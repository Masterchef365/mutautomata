//use rand::Rng;
//use rand::SeedableRng;
//use rand::rngs::SmallRng;
use rand::prelude::*;
use watertender::trivial::*;
use watertender::vertex::Vertex;

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

struct Step {
    pos: [i32; 3],
    color: u8,
}

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
        const W: i32 = 8000;
        self.pos[0] = self.pos[0] % W;
        self.pos[1] = self.pos[1] % W;
        self.pos[2] = self.pos[2] % W;
        */

        Some(self.do_plot.then(|| Step { pos: self.pos, color: self.color }))
    }
}

fn decode(v: &[u8]) -> Vec<Instruction> {
    v.iter().map(|&i| i.into()).collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
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

    println!("Using seed {}", seed);
    
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

    let max_steps_per_object = 30_000;

    //let radius = 10.;
    //let pos = [0.; 3];
    //let cost_fn = |pt: [f32; 3]| sphere_cost(pt, pos, radius);
    let size = 100.;
    let cost_fn = |pos: [f32; 3]| sine_cost(pos, size);
    let gene_pool = evolution(&mut rng, cost_fn, initial_dir, initial_pos, max_steps_per_object);

    let vertex_budget = 300_000;

    let mut objects = vec![];
    let mut total_vertices = 0;
    for code in gene_pool {
        let code = decode(&code);
        let state = State::new(code, initial_dir, initial_pos);
        let steps = eval(state, max_steps_per_object);
        let geom = plot_lines(&steps, mode);
        total_vertices += geom.vertices.len();
        if total_vertices > vertex_budget {
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

fn evolution(rng: &mut impl Rng, cost_fn: impl Fn([f32; 3]) -> f32, initial_dir: Direction, initial_pos: [i32; 3], max_steps_per_object: usize) -> Vec<Vec<u8>> {
    let code_length = 8_000;

    let code: Vec<u8> = (0..code_length).map(|_| rng.gen()).collect();

    let n_offspring = 2; // And one more, which is just the original!
    let n_kept = 10; // Keep up to this many after evaluations
    let n_generations = 1000;

    let compute_code_cost = |code: &[u8]| {
        let instructions = decode(&code);
        let state = State::new(instructions, initial_dir, initial_pos);
        let steps = eval(state, max_steps_per_object);
        let mut cost = compute_cost(&steps, &cost_fn);
        cost
    };
    
    let cost = compute_code_cost(&code);
    let mut gene_pool: Vec<(Vec<u8>, f32)> = vec![(code, cost)];

    let max_mutations = 100;

    for gen_idx in 1..=n_generations {
        println!("Computing generation {}, starting with {} gene sets", gen_idx, gene_pool.len());

        let mut new_genes = vec![];
        for gene_set in &gene_pool {
            for _ in 0..n_offspring {
                // Create mutations
                let (mut code, _) = gene_set.clone();
                for _ in 0..rng.gen_range(1..=max_mutations) {
                    *code.choose_mut(rng).unwrap() = rng.gen::<u8>().into();
                }

                // Evaluate and add to gene pool
                let cost = compute_code_cost(&code);
                new_genes.push((code, cost));
            }
        }

        gene_pool.extend(new_genes);

        // Sort by best cost descending
        gene_pool.sort_by(|(_, cost_a), (_, cost_b)| 
            cost_a
                .partial_cmp(cost_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        );

        dbg!(gene_pool.first().unwrap().1);

        let saved = gene_pool.choose_multiple(rng, 5).cloned().collect::<Vec<_>>();
        gene_pool.truncate(n_kept);
        gene_pool.extend(saved);
    }

    gene_pool.into_iter().take(n_kept).map(|(code, _cost)| code).collect()
}

fn sphere_cost(point: [f32; 3], pos: [f32; 3], radius: f32) -> f32 {
    let [x, y, z] = point;
    let [px, py, pz] = pos;
    let squared_dist = (x - px).powf(2.) + (y - py).powf(2.) + (z - pz).powf(2.);
    let signed_sq_dist = squared_dist - radius * radius;
    return signed_sq_dist;
}

fn cube_cost([x, y, z]: [f32; 3], size: f32) -> f32 {
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
    let scale = |v: i32| v as f32 / 100.;

    let vertices = steps
        .iter()
        .map(|&Step { pos: [x, y, z], color }| Vertex::new([scale(x), scale(y), scale(z)], color_lut(color)))
        .collect::<Vec<Vertex>>();

    match mode {
        PlotMode::Lines => DrawData {
            indices: (1u32..).take((vertices.len() - 1) * 2).map(|i| i / 2).collect(),
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
