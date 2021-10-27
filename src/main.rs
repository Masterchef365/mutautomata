//use rand::Rng;
//use rand::SeedableRng;
//use rand::rngs::SmallRng;
use rand::prelude::*;
use watertender::trivial::*;
use watertender::vertex::Vertex;
use structopt::StructOpt;

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
    /// Coordinate wraparound, if any
    wrap: Option<usize>,
}

impl State {
    fn new(code: Vec<Instruction>, dir: Direction, pos: [i32; 3], wrap: Option<usize>) -> Self {
        Self {
            code,
            dir,
            color: 0,
            pos,
            do_plot: true,
            ip: 0,
            max_ip: 0,
            stack: vec![],
            wrap,
        }
    }
}

impl Iterator for State {
    type Item = Option<([i32; 3], u8)>;
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

        if let Some(wrap) = self.wrap {
            let wrap = wrap as i32;
            let wrap = |pos: i32| if pos.abs() > wrap { 
                if pos > 0 {
                    pos - wrap
                } else {
                    pos + wrap
                }
            } else {
                pos
            };
            self.pos[0] = wrap(self.pos[0]);
            self.pos[1] = wrap(self.pos[1]);
            self.pos[2] = wrap(self.pos[2]);
        }

        Some(self.do_plot.then(|| (self.pos, self.color)))
    }
}

fn decode(v: &[u8]) -> Vec<Instruction> {
    v.iter().map(|&i| i.into()).collect()
}

#[derive(Debug, StructOpt)]
#[structopt(name = "example", about = "An example of StructOpt usage.")]
struct Opt {
    /// Maximum program length
    #[structopt(short = "l", long, default_value = "8000")]
    code_length: i32,

    /// Maximum number of vertices to be produced
    #[structopt(short, long, default_value = "3000000")]
    vertex_budget: usize,

    /// Maximum number of program steps per run
    #[structopt(short = "p", long, default_value = "300000")]
    max_steps_per_object: usize,

    /// Maximum number of mutations
    #[structopt(short = "u", long, default_value = "100")]
    max_mutations: i32,

    /// RNG seed
    #[structopt(short, long)]
    seed: Option<u64>,

    /// Print the instructions used
    #[structopt(long)]
    show_instructions: bool,

    /// The primitive to use (one of lines, triangles, or points)
    #[structopt(short = "p", long, default_value = "lines")]
    primitive: PlotMode,

    /// Use vr!
    #[structopt(short = "r", long)]
    vr: bool,

    /// Wrap coordinates by this value
    #[structopt(short, long)]
    wrap: Option<usize>,
}

fn main() {
    let args = Opt::from_args();

    let seed = args.seed.unwrap_or_else(|| rand::thread_rng().gen());
    let mut rng = SmallRng::seed_from_u64(seed);
    println!("Using seed {}", seed);

    let code: Vec<u8> = (0..args.code_length).map(|_| rng.gen()).collect();
    let mut code = decode(&code);

    if args.show_instructions {
        for (ip, text) in code.iter().enumerate() {
            println!("{}: {:?}", ip, text);
        }
    }

    let mut objects = vec![];

    let mut total_vertices = 0;
    while total_vertices < args.vertex_budget {
        let initial_dir = match rng.gen::<u32>() % 6 {
            0 => Direction::X,
            1 => Direction::Y,
            2 => Direction::Z,
            3 => Direction::NegX,
            4 => Direction::NegY,
            _ => Direction::NegZ,
        };

        let mut state = State::new(code.clone(), initial_dir, [0; 3], args.wrap);

        /*
        for (step, [x, y, z]) in state.take(80).enumerate() {
            println!("{:>5}: {}, {}, {}", step, x, y, z);
        }
        */

        let pcld = plot_lines(&mut state, args.vertex_budget.min(args.max_steps_per_object), args.primitive);
        if !pcld.indices.is_empty() && !pcld.vertices.is_empty() {
            total_vertices += pcld.vertices.len();
            objects.push(pcld);
        }

        // Mutate code
        for _ in 0..rng.gen_range(1..=args.max_mutations) {
            *code.choose_mut(&mut rng).unwrap() = rng.gen::<u8>().into();
        }
    }

    draw(objects, args.vr).expect("Draw failed");
}

#[derive(Copy, Clone, Debug)]
enum PlotMode {
    Lines,
    Points,
    Triangles,
}

impl std::str::FromStr for PlotMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.to_lowercase().as_str() {
            "points" | "point" | "p" => Self::Points,
            "triangles" | "triangle" | "tri" | "t" => Self::Triangles,
            "lines" | "line" | "l" => Self::Lines,
            other => return Err(format!("{} is not one of points, lines, or triangles.", other)),
        })
    }
}

fn plot_lines(state: &mut State, n: usize, mode: PlotMode) -> DrawData {
    let scale = |v: i32| v as f32 / 100.;

    let vertices = state
        .take(n)
        .filter_map(|c| c)
        .map(|([x, y, z], c)| Vertex::new([scale(x), scale(y), scale(z)], color_lut(c)))
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
        [0xff, 0, 0],
        [0, 0xff, 0],
        [0, 0, 0xff],
        [0, 0xff, 0xff],
        [0xff, 0xff, 0],
        [0xff, 0, 0xff],
    ];

    let [r, g, b] = COLORS[v as usize % COLORS.len()];
    let s = |c: u8| c as f32 / 255.;
    [s(r), s(g), s(b)]
}
