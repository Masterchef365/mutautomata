//use rand::Rng;
//use rand::SeedableRng;
//use rand::rngs::SmallRng;
use rand::prelude::*;
use rayon::prelude::*;
use watertender::trivial::*;
use watertender::vertex::Vertex;
use weezl::{encode::Encoder, BitOrder};

const SCALE: i32 = 100;

#[derive(Debug, Copy, Clone)]
enum Instruction {
    /// Turn to the specified direction
    Turn(Direction),
    /// Push a repeat onto the stack
    Repeat(u8),
    /// Set the color
    Color(u8),
    /// Plot vertex
    Plot,
    /// Only step
    Noop,
    /// Jump to just after the last repeat instruction or pass through if we've exhausted it
    Jump,
}

#[derive(Debug, Copy, Clone)]
enum Direction {
    X,
    Y,
    Z,
    NegX,
    NegY,
    NegZ,
}

impl Into<[i32; 3]> for Direction {
    /// Decode a direction
    fn into(self) -> [i32; 3] {
        match self {
            Direction::X => [1, 0, 0],
            Direction::Y => [0, 1, 0],
            Direction::Z => [0, 0, 1],
            Direction::NegX => [-1, 0, 0],
            Direction::NegY => [0, -1, 0],
            Direction::NegZ => [0, 0, -1],
        }
    }
}

impl From<u8> for Instruction {
    /// Decode an instruction
    fn from(v: u8) -> Self {
        match v % 18 {
            0 | 1 => Self::Turn(Direction::X),
            2 | 3 => Self::Turn(Direction::Y),
            4 | 5 => Self::Turn(Direction::Z),
            6 | 7 => Self::Turn(Direction::NegX),
            8 | 9 => Self::Turn(Direction::NegY),
            10 | 11 => Self::Turn(Direction::NegZ),
            12 => Self::Jump,
            14 => Self::Plot,
            15 | 16 => Self::Repeat(v / 64),
            _ => Self::Color(v % 16),
            //_ => Self::Noop,
        }
    }
}

#[derive(Clone)]
struct State {
    /// Instructions
    code: Vec<Instruction>,
    /// Direction
    dir: Direction,
    /// Position as [x, y, z]
    pos: [i32; 3],
    /// Color
    color: u8,
    /// Instruction pointer
    ip: usize,
    /// Stack; consists of 'pointer' and 'remaining repeats'
    stack: Vec<(usize, u8)>,
    /// Maximum location of the instruction pointser
    max_ip: usize,
    /// Whether or not to plot
    do_plot: bool,
}

impl State {
    fn new(code: Vec<Instruction>, dir: Direction, pos: [i32; 3]) -> Self {
        Self {
            code,
            dir,
            color: 0,
            pos,
            do_plot: true,
            ip: 0,
            max_ip: 0,
            stack: vec![],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Step {
    pub pos: [i32; 3],
    pub color: u8,
}

unsafe impl bytemuck::Zeroable for Step {}
unsafe impl bytemuck::Pod for Step {}

impl Iterator for State {
    type Item = Option<Step>;
    /// Machine step
    fn next(&mut self) -> Option<Self::Item> {
        let inst = *self.code.get(self.ip)?;

        match inst {
            Instruction::Turn(dir) => self.dir = dir,
            Instruction::Plot => self.do_plot = !self.do_plot,
            Instruction::Repeat(n) => self.stack.push((self.ip, n)),
            Instruction::Noop => (),
            Instruction::Jump => match self.stack.pop() {
                Some((last_ptr, last_count)) if last_count > 0 => {
                    self.ip = last_ptr;
                    self.stack.push((self.ip, last_count - 1))
                }
                _ => (),
            },
            Instruction::Color(c) => self.color = c,
        }

        // Advance instruction pointer
        self.ip += 1;
        self.max_ip = self.max_ip.max(self.ip);

        // Step in the direction
        let [dx, dy, dz]: [i32; 3] = self.dir.into();
        self.pos[0] += dx;
        self.pos[1] += dy;
        self.pos[2] += dz;

        /*
        const W: i32 = 1000;
        self.pos[0] = self.pos[0] % W;
        self.pos[1] = self.pos[1] % W;
        self.pos[2] = self.pos[2] % W;
        */

        Some(self.do_plot.then(|| Step {
            pos: self.pos,
            color: self.color,
        }))
    }
}

fn decode(v: &[u8]) -> Vec<Instruction> {
    v.iter().map(|&i| i.into()).collect()
}

fn sigmoid(v: f32) -> f32 {
    1. / (1. + (-v).exp())
}

fn inside_aabb(pos: [i32; 3], min: [i32; 3], max: [i32; 3]) -> bool {
    pos.into_iter()
        .zip(min)
        .zip(max)
        .all(|((&p, i), a)| p >= i && p < a)
}

fn aabb_center([ix, iy, iz]: [i32; 3], [ax, ay, az]: [i32; 3]) -> [i32; 3] {
    [
        (ix + ax) / 2,
        (iy + ay) / 2,
        (iz + az) / 2,
    ]
}

fn distance_sq([px, py, pz]: [i32; 3], [qx, qy, qz]: [i32; 3]) -> i32 {
    let dx = px - qx;
    let dy = py - qy;
    let dz = pz - qz;
    dx * dx + dy * dy + dz * dz
}

fn main() {
    let mut args = std::env::args().skip(1);
    let image_path = args.next().expect("Requires image path");

    let decoder = std::fs::File::open(image_path).expect("Image open failed");
    let decoder = png::Decoder::new(decoder);
    let mut reader = decoder.read_info().unwrap();
    // Allocate the output buffer.
    let mut buf = vec![0; reader.output_buffer_size()];
    // Read the next frame. An APNG might contain multiple frames.
    let info = reader.next_frame(&mut buf).unwrap();

    assert_eq!(info.color_type, png::ColorType::Grayscale);
    assert_eq!(info.bit_depth, png::BitDepth::Eight);

    let width = 10;

    let mode = args.next();
    let seed = args.next();

    let mode = match mode.as_ref().map(|s| s.as_str()) {
        Some("points") => PlotMode::Points,
        Some("triangles" | "tri") => PlotMode::Triangles,
        _ => PlotMode::Lines,
    };

    let seed = match seed {
        None => rand::thread_rng().gen(),
        Some(s) => s.parse::<u64>().expect("Failed to parse seed"),
    };

    eprintln!("Using seed {}", seed);

    let mut rng = SmallRng::seed_from_u64(seed);

    let initial_dir = match rng.gen::<u32>() % 6 {
        0 => Direction::X,
        1 => Direction::Y,
        2 => Direction::Z,
        3 => Direction::NegX,
        4 => Direction::NegY,
        _ => Direction::NegZ,
    };
    let initial_pos = [0; 3];

    let max_steps_per_object = 100_000;

    // Use the compression ratio on the position sequence to determine the score! (Usinge
    // DEFLATE or some shit)

    let bin_size = 100;

    let image_data = image_plane(&buf, info.width as _, bin_size as f32 / SCALE as f32);

    let cost_fn = move |steps: &[Step]| {
        let mut map = std::collections::HashSet::new();

        let min = [0; 3];
        let max = [
            info.width as i32 * bin_size, 
            bin_size, 
            info.height as i32 * bin_size
        ];

        let center = aabb_center(min, max);

        let mut cost = 0.0;

        for step in steps {
            if !map.insert(step.pos) {
                cost += 1.0;
            }

            if !inside_aabb(step.pos, min, max) {
                cost += (distance_sq(center, step.pos).max(1) as f32).ln();
            } else {
                let [x, _, z] = step.pos;
                let x = x / bin_size;
                let y = z / bin_size;

                let bin = buf[y as usize * info.width as usize + x as usize];
                cost -= bin as f32 / 255.0;
            }
        }

        cost
    };
    let gene_pool = evolution(cost_fn, initial_dir, initial_pos, max_steps_per_object);

    let vertex_budget = 3_000_000;

    let mut objects = vec![image_data];
    let mut total_vertices = 0;
    for code in gene_pool {
        let code = decode(&code);
        let state = State::new(code, initial_dir, initial_pos);
        let steps = eval(state, max_steps_per_object);
        let geom = plot_lines(&steps, mode);
        total_vertices += geom.vertices.len();
        if dbg!(total_vertices) > vertex_budget {
            break;
        }
        objects.push(geom);
    }

    let vr = std::env::var("MUT_VR").is_ok();

    draw(objects, vr).expect("Draw failed");
}

#[derive(Copy, Clone)]
enum PlotMode {
    Lines,
    Points,
    Triangles,
}

fn evolution(
    cost_fn: impl Fn(&[Step]) -> f32 + Sync,
    initial_dir: Direction,
    initial_pos: [i32; 3],
    max_steps_per_object: usize,
) -> Vec<Vec<u8>> {
    let mut rng = rand::thread_rng();

    let code_length = 8_000;
    let n_offspring = 6; // And one more, which is just the original!
    let n_kept = 45; // Keep up to this many after evaluations
    let n_generations = 400;
    let max_mutations = 150;

    let code: Vec<u8> = (0..code_length).map(|_| rng.gen()).collect();

    let compute_code_cost = |code: &[u8]| {
        let instructions = decode(&code);
        let state = State::new(instructions, initial_dir, initial_pos);
        let steps = eval(state, max_steps_per_object);
        cost_fn(&steps)
    };

    let cost = compute_code_cost(&code);
    let mut gene_pool: Vec<(Vec<u8>, f32)> = vec![(code, cost)];

    for gen_idx in 1..=n_generations {
        eprintln!(
            "Computing generation {}, starting with {} gene sets",
            gen_idx,
            gene_pool.len()
        );

        let new_genes = gene_pool
            .par_iter()
            .map(|gene_set| {
                let mut rng = rand::thread_rng();
                let mut new_genes = vec![];
                for _ in 0..n_offspring {
                    // Create mutations
                    let (mut code, _) = gene_set.clone();
                    for _ in 0..rng.gen_range(1..=max_mutations) {
                        *code.choose_mut(&mut rng).unwrap() = rng.gen::<u8>().into();
                    }

                    // Evaluate and add to gene pool
                    let cost = compute_code_cost(&code);
                    new_genes.push((code, cost));
                }
                new_genes
            })
            .flatten()
            .collect::<Vec<_>>();

        gene_pool.extend(new_genes);

        // Sort by best cost descending
        gene_pool.sort_by(|(_, cost_a), (_, cost_b)| {
            cost_a
                .partial_cmp(cost_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let lowest_cost = gene_pool.first().unwrap().1;
        dbg!(lowest_cost);

        let saved = gene_pool
            .choose_multiple(&mut rng, 5)
            .cloned()
            .collect::<Vec<_>>();
        gene_pool.truncate(n_kept);
        gene_pool.extend(saved);
    }

    gene_pool
        .into_iter()
        .take(n_kept)
        .map(|(code, _cost)| code)
        .collect()
}

/*
fn sphere_cost(steps: &[Step], pos: [f32; 3], radius: f32) -> f32 {
    steps.iter().map(|s| sphere_pt_cost([(s.pos[0] - pos[0]) as f32, s.pos[1] as f32, s.pos[2] as f32], [0.; 3], radius)).sum::<f32>()
}

fn sphere_pt_cost(point: [f32; 3], pos: [f32; 3], radius: f32) -> f32 {
    let [x, y, z] = point;
    let [px, py, pz] = pos;
    let squared_dist = (x - px).powf(2.) + (y - py).powf(2.) + (z - pz).powf(2.);
    let signed_sq_dist = squared_dist - radius * radius;
    if signed_sq_dist < 0. {
        0.
    } else {
        signed_sq_dist
    }
}
*/

fn add_n<T, const N: usize>(x: [T; N], y: [T; N]) -> [T; N]
where
    T: std::ops::Add<Output = T> + Default + Copy,
{
    let mut output = [T::default(); N];
    output
        .iter_mut()
        .zip(&x)
        .zip(&y)
        .for_each(|((o, x), y)| *o = *x + *y);
    output
}

fn neg_n<T, const N: usize>(x: [T; N]) -> [T; N]
where
    T: std::ops::Neg<Output = T> + Default + Copy,
{
    let mut output = [T::default(); N];
    output.iter_mut().zip(&x).for_each(|(o, x)| *o = -*x);
    output
}

fn dot_n<T, const N: usize>(x: [T; N], y: [T; N]) -> T
where
    T: std::iter::Sum + std::ops::Mul<Output = T> + Copy,
{
    x.iter().zip(&y).map(|(x, y)| *x * *y).sum::<T>()
}

fn div_n<T, const N: usize>(x: [T; N], y: T) -> [T; N]
where
    T: std::iter::Sum + std::ops::Div<Output = T>,
    T: Default + Copy,
{
    let mut output = [T::default(); N];
    output.iter_mut().zip(&x).for_each(|(o, x)| *o = *x / y);
    output
}

fn cast_n<T, U, const N: usize>(x: [T; N]) -> [U; N]
where
    T: Into<U> + Copy,
    U: Default + Copy,
{
    let mut output = [U::default(); N];
    output
        .iter_mut()
        .zip(&x)
        .for_each(|(o, x)| *o = (*x).into());
    output
}

fn cast_3_i32_f32([x, y, z]: [i32; 3]) -> [f32; 3] {
    [x as f32, y as f32, z as f32]
}

fn step_pos_stddev(steps: &[Step]) -> f32 {
    let sum = steps.iter().fold([0; 3], |sum, step| add_n(sum, step.pos));
    let mean = div_n(sum, steps.len() as i32);
    let variance = steps
        .iter()
        .map(|step| {
            let diff = add_n(step.pos, neg_n(mean));
            dot_n(diff, diff) as u64
        })
        .sum::<u64>() as f32
        / steps.len() as f32;
    let std_dev = variance.sqrt();
    std_dev
}

fn cube_cost(steps: &[Step], size: f32) -> f32 {
    steps
        .iter()
        .map(|s| cube_pt_cost(cast_3_i32_f32(s.pos), size))
        .sum::<f32>()
}

fn cube_pt_cost([x, y, z]: [f32; 3], size: f32) -> f32 {
    let in_bounds = |v: f32| v >= -size && v <= size;
    if in_bounds(x) && in_bounds(y) && in_bounds(z) {
        //let dist_sq = (x).powf(2.) + (y).powf(2.) + (z).powf(2.);
        //size * size * size - dist_sq
        0.0
    } else {
        let clamp = |v: f32| v.max(-size).min(size);
        let [nx, ny, nz] = [clamp(x), clamp(y * 8.), clamp(z)];
        (x - nx).powf(2.) + (y - ny).powf(2.) + (z - nz).powf(2.)
    }
}

fn sine_cost([x, y, z]: [f32; 3], size: f32) -> f32 {
    let h = (x / 100.).cos() + (z / 100.).sin();
    (h * size - y).abs()
}

fn compute_cost(steps: &[Step], cost_fn: impl Fn([f32; 3]) -> f32) -> f32 {
    let mut cost = 0.;
    for step in steps {
        let [x, y, z] = step.pos;
        let pos = [x as f32, y as f32, z as f32];
        cost += cost_fn(pos);
    }
    cost
}

fn eval(state: State, max_steps: usize) -> Vec<Step> {
    state.take(max_steps).filter_map(|c| c).collect()
}

fn plot_lines(steps: &[Step], mode: PlotMode) -> DrawData {
    let scale = |v: i32| v as f32 / SCALE as f32;

    let vertices = steps
        .iter()
        .map(
            |&Step {
                 pos: [x, y, z],
                 color,
             }| Vertex::new([scale(x), scale(y), scale(z)], color_lut(color)),
        )
        .collect::<Vec<Vertex>>();

    match mode {
        PlotMode::Lines => DrawData {
            indices: (1u32..)
                .take((vertices.len() - 1) * 2)
                .map(|i| i / 2)
                .collect(),
            vertices,
            primitive: Primitive::Lines,
        },
        PlotMode::Points => DrawData {
            indices: (0..vertices.len() as u32).collect(),
            vertices,
            primitive: Primitive::Points,
        },
        PlotMode::Triangles => DrawData {
            indices: (0..vertices.len() as u32).collect(),
            vertices,
            primitive: Primitive::Triangles,
        },
    }
}

fn color_lut(v: u8) -> [f32; 3] {
    /*
    const colors: [[u8; 3]; 7] = [
        [0xff; 3],
        [165, 66, 66],
        [140, 148, 64],
        [222, 147, 95],
        [95, 129, 157],
        [133, 103, 143],
        [94, 141, 135],
    ];
    */

    const COLORS: [[u8; 3]; 7] = [
        [0xff; 3],
        [0xff, 0x44, 0],
        [0x44, 0xff, 0],
        [0, 0x44, 0xff],
        [0x44, 0xff, 0xff],
        [0xff, 0xff, 0],
        [0xff, 0, 0xff],
    ];

    let [r, g, b] = COLORS[v as usize % COLORS.len()];
    let s = |c: u8| c as f32 / 255.;
    [s(r), s(g), s(b)]
}

fn image_plane(buf: &[u8], width: usize, size: f32) -> DrawData {
    assert_eq!(buf.len() % width, 0);
    let height = buf.len() / width;

    let mut vertices = vec![];
    let mut indices: Vec<u32> = vec![];

    let mut idx = 0;

    for row in 0..height {
        for col in 0..height {
            let pixel = buf[idx];

            let color = [pixel as f32 / 255.; 3];

            let x = col as f32 * size;
            let z = row as f32 * size;

            let base = vertices.len() as u32;
            vertices.push(Vertex::new([x, 0., z], color));
            vertices.push(Vertex::new([x + size, 0., z], color));
            vertices.push(Vertex::new([x + size, 0., z + size], color));
            vertices.push(Vertex::new([x, 0., z + size], color));

            indices.extend([3, 1, 0, 3, 2, 1].iter().map(|i| i + base));

            idx += 1;
        }
    }

    DrawData {
        indices,
        vertices,
        primitive: Primitive::Triangles,
    }
}
