use watertender::trivial::*;
use watertender::vertex::Vertex;
use rand::Rng;

#[derive(Debug, Copy, Clone)]
enum Instruction {
    /// Turn to the specified direction
    Turn(Direction),
    /// Push a repeat onto the stack
    Repeat(u8),
    /// Set the color
    Color(u8),
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
        match v / 16 {
            0 => Self::Turn(Direction::X),
            1 => Self::Turn(Direction::Y),
            2 => Self::Turn(Direction::Z),
            3 => Self::Turn(Direction::NegX),
            4 => Self::Turn(Direction::NegY),
            5 => Self::Turn(Direction::NegZ),
            6 => Self::Color(v % 16),
            7..=9 => Self::Repeat(v % 2),
            10..=12 => Self::Jump,
            _ => Self::Noop,
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
}

impl State {
    fn new(code: Vec<Instruction>, dir: Direction, pos: [i32; 3]) -> Self {
        Self {
            code,
            dir,
            color: 0,
            pos,
            ip: 0,
            stack: vec![],
        }
    }
}

impl Iterator for State {
    type Item = ([i32; 3], u8);
    /// Machine step
    fn next(&mut self) -> Option<Self::Item> {
        let inst = *self.code.get(self.ip)?;

        match inst {
            Instruction::Turn(dir) => self.dir = dir,
            Instruction::Noop => (),
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

        // Step in the direction
        let [dx, dy, dz]: [i32; 3] = self.dir.into();
        self.pos[0] += dx;
        self.pos[1] += dy;
        self.pos[2] += dz;

        const W: i32 = 1500;
        self.pos[0] = self.pos[0] % W;
        self.pos[1] = self.pos[1] % W;
        self.pos[2] = self.pos[2] % W;

        Some((self.pos, self.color))
    }
}

fn decode(v: &[u8]) -> Vec<Instruction> {
    v.iter().map(|&i| i.into()).collect()
}

fn main() {
    let arg = std::env::args().skip(1).next();
    let mode = match arg.as_ref().map(|s| s.as_str()) {
        Some("points") => PlotMode::Points,
        Some("triangles" | "tri") => PlotMode::Triangles,
        _ => PlotMode::Lines,
    };

    let mut rng = rand::thread_rng();
    let code: Vec<u8> = (0..2000).map(|_| rng.gen()).collect();
    let code = decode(&code);
    
    /*
    let code = vec![
        Instruction::Repeat(2),
        Instruction::Turn(Direction::NegZ),
        Instruction::Repeat(2),
        Instruction::Turn(Direction::NegX),
        Instruction::Jump,
        Instruction::Turn(Direction::NegY),
        Instruction::Jump,
    ];
    */

    for (ip, text) in code.iter().enumerate() {
        println!("{}: {:?}", ip, text);
    }

    let state = State::new(code, Direction::X, [0; 3]);

    /*
    for (step, [x, y, z]) in state.take(80).enumerate() {
        println!("{:>5}: {}, {}, {}", step, x, y, z);
    }
    */

    let pcld = plot_lines(state, 30_000_000, mode);
    dbg!(pcld.vertices.len());
    draw(vec![pcld], false).expect("Draw failed");
}

enum PlotMode {
    Lines,
    Points,
    Triangles,
}

fn plot_lines(state: State, n: usize, mode: PlotMode) -> DrawData {

    let scale = |v: i32| v as f32 / 100.;

    let vertices = state
        .take(n)
        .map(|([x, y, z], c)| 
            Vertex::new([
                scale(x), 
                scale(y), 
                scale(z)
            ], color_lut(c))
        )
        .collect::<Vec<Vertex>>();

    match mode {
        PlotMode::Lines => {
            DrawData {
                indices: (1u32..).take(vertices.len() * 2).map(|i| i / 2).collect(),
                vertices,
                primitive: Primitive::Lines,
            }
        }
        PlotMode::Points => {
            DrawData {
                indices: (0..vertices.len() as u32).collect(),
                vertices,
                primitive: Primitive::Points,
            }
        }
        PlotMode::Triangles => {
            DrawData {
                indices: (0..vertices.len() as u32).collect(),
                vertices,
                primitive: Primitive::Triangles,
            }
        }
    }
}

fn color_lut(v: u8) -> [f32; 3] {
    const COLORS: [[u8; 3]; 7] = [
        [0xff; 3],
        [165, 66, 66],
        [140, 148, 64],
        [222, 147, 95],
        [95, 129, 157],
        [133, 103, 143],
        [94, 141, 135],
    ];
    let [r, g, b] = COLORS[v as usize % COLORS.len()];
    let s = |c: u8| c as f32 / 255.;
    [s(r), s(g), s(b)]
}
