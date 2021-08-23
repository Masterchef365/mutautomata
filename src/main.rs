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

impl Iterator for State {
    type Item = Option<([i32; 3], u8)>;
    /// Machine step
    fn next(&mut self) -> Option<Self::Item> {
        let inst = *self.code.get(self.ip)?;

        match inst {
            Instruction::Turn(dir) => self.dir = dir,
            Instruction::Plot => self.do_plot = !self.do_plot,
            Instruction::Repeat(n) => self.stack.push((self.ip, n)),
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

        Some(self.do_plot.then(|| (self.pos, self.color)))
    }
}

fn decode(v: &[u8]) -> Vec<Instruction> {
    v.iter().map(|&i| i.into()).collect()
}

#[derive(Debug, StructOpt)]
#[structopt(name = "MutAutomata", about = "Builder of spindly structures and such")]
struct Opt {
    /// Size of the genome
    #[structopt(short, long, default_value="8000")]
    genome_length: usize,

    /// Max vertices to display
    #[structopt(short, long, default_value="3000000")]
    vertex_budget: usize,

    /// Max steps which can be taken for rendering each shape
    #[structopt(short = "m", long, default_value="300000")]
    max_steps_per_object: usize,

    /// Maximum number of mutations per child
    #[structopt(short = "u", long, default_value="100")]
    max_mutations: usize,

    /// Plot using this primitive/mode
    #[structopt(short, long, default_value="lines")]
    plot_mode: PlotMode,

    /// Show instructions of the seed
    #[structopt(long)]
    show_instructions: bool,

    /// Supply a seed
    #[structopt(short, long)]
    seed: Option<u64>,

    /// Maximum number of colors
    #[structopt(long, default_value="10")]
    max_colors: u32, 

    #[structopt(long, default_value="0.0")]
    min_bright: f32, 

    #[structopt(long, default_value="1.0")]
    max_bright: f32
}

/*
#[derive(Debug, StructOpt)]
#[structopt(name = "MutAutomata", about = "Builder of spindly structures and such")]
struct rOpt {
    /// Activate debug mode
    // short and long flags (-d, --debug) will be deduced from the field's name
    #[structopt(short, long)]
    debug: bool,

    /// Set speed
    // we don't want to name it "speed", need to look smart
    #[structopt(short = "v", long = "velocity", default_value = "42")]
    speed: f64,

    /// Input file
    #[structopt(parse(from_os_str))]
    input: PathBuf,

    /// Output file, stdout if not present
    #[structopt(parse(from_os_str))]
    output: Option<PathBuf>,

    /// Where to write the output: to `stdout` or `file`
    #[structopt(short)]
    out_type: String,

    /// File name: only required when `out-type` is set to `file`
    #[structopt(name = "FILE", required_if("out-type", "file"))]
    file_name: Option<String>,
}
*/

fn main() {
    let opt = Opt::from_args();

    let seed = opt.seed.unwrap_or(rand::thread_rng().gen());
    println!("Using seed {}", seed);

    let mut rng = SmallRng::seed_from_u64(seed);

    let code: Vec<u8> = (0..opt.genome_length).map(|_| rng.gen()).collect();
    let mut code = decode(&code);

    if opt.show_instructions {
        for (ip, text) in code.iter().enumerate() {
            println!("{}: {:?}", ip, text);
        }
    }

    let mut objects = vec![];

    let colors = random_color_lut(&mut rng, opt.max_colors, opt.min_bright, opt.max_bright);

    let mut total_vertices = 0;
    while total_vertices < opt.vertex_budget {
        let initial_dir = match rng.gen::<u32>() % 6 {
            0 => Direction::X,
            1 => Direction::Y,
            2 => Direction::Z,
            3 => Direction::NegX,
            4 => Direction::NegY,
            _ => Direction::NegZ,
        };

        let mut state = State::new(code.clone(), initial_dir, [0; 3]);

        let n_verts = opt.vertex_budget.min(opt.max_steps_per_object);
        let pcld = plot_lines(&mut state, n_verts, opt.plot_mode, &colors);
        if !pcld.indices.is_empty() && !pcld.vertices.is_empty() {
            total_vertices += pcld.vertices.len();
            objects.push(pcld);
        }

        // Mutate code
        for _ in 0..rng.gen_range(1..=opt.max_mutations) {
            *code.choose_mut(&mut rng).unwrap() = rng.gen::<u8>().into();
        }
    }

    let vr = std::env::var("MUT_VR").is_ok();
    draw(objects, vr).expect("Draw failed");
}

#[derive(Debug, Copy, Clone)]
enum PlotMode {
    Lines,
    Points,
    Triangles,
}

use std::str::FromStr;
impl FromStr for PlotMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "lines" | "l" => Ok(Self::Lines),
            "points" | "p" => Ok(Self::Points),
            "triangles" | "tris" | "t" => Ok(Self::Triangles),
            _ => Err(format!("\"s\" did not match lines|points|tris")),
        }
    }
}

fn plot_lines(state: &mut State, n: usize, mode: PlotMode, colors: &[[f32; 3]]) -> DrawData {
    let scale = |v: i32| v as f32 / 100.;

    let vertices = state
        .take(n)
        .filter_map(|c| c)
        .map(|([x, y, z], c)| Vertex::new([scale(x), scale(y), scale(z)], colors[c as usize % colors.len()]))
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

fn random_color_lut(
    rng: &mut impl Rng, 
    max_colors: u32, 
    min_bright: f32, 
    max_bright: f32
) -> Vec<[f32; 3]> {
    let mut colors = vec![];
    for _ in 0..rng.gen_range(0..max_colors) {
        let mut color = [
            rng.gen_range(min_bright..max_bright),
            rng.gen_range(min_bright..max_bright),
            rng.gen_range(min_bright..max_bright),
        ];
        let idx = rng.gen_range(0..color.len());
        color[idx] = 0.;
        colors.push(color);
    }
    colors
}
