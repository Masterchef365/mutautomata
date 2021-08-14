#[derive(Debug, Copy, Clone)]
enum Instruction {
    /// Turn to the specified direction
    Turn(Direction),
    /// Jump to just after the last repeat instruction or the beginning n times
    Repeat(u8),
    /// Only step
    Noop,
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
            6 => {
                let rem = v % 16;
                if rem < 2 {
                    Self::Repeat(rem)
                } else {
                    Self::Noop
                }
            }
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
            pos,
            ip: 0,
            stack: vec![],
        }
    }
}

impl Iterator for State {
    type Item = [i32; 3];
    /// Machine step
    fn next(&mut self) -> Option<Self::Item> {
        let inst = *self.code.get(self.ip)?;

        match inst {
            Instruction::Turn(dir) => self.dir = dir,
            Instruction::Noop => (),
            Instruction::Repeat(times) => match self.stack.pop() {
                Some((last_ptr, last_count)) if last_count > 0 => {
                    self.ip = last_ptr;
                    self.stack.push((self.ip, last_count - 1))
                }
                _ => self.stack.push((self.ip, times)),
            },
        }

        // Advance instruction pointer
        self.ip += 1;

        // Step in the direction
        let [dx, dy, dz]: [i32; 3] = self.dir.into();
        self.pos[0] += dx;
        self.pos[1] += dy;
        self.pos[2] += dz;

        Some(self.pos)
    }
}

fn decode(v: &[u8]) -> Vec<Instruction> {
    v.iter().map(|&i| i.into()).collect()
}

fn main() {
    let arg = std::env::args().skip(1).next().unwrap_or("Hello, world!".into());
    let code = decode(arg.as_bytes());
    dbg!(&code);

    let state = State::new(code, Direction::X, [0; 3]);
    for (step, [x, y, z]) in state.take(80).enumerate() {
        println!("{:>5}: {}, {}, {}", step, x, y, z);
    }
}
