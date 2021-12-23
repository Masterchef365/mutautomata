use structopt::StructOpt;
use mutautomata::DrawData;
mod draw;

#[derive(Debug, StructOpt)]
#[structopt(name = "MutAutomata", about = "Builder of spindly structures and such")]
struct Opt {
    /// Enable VR view
    #[structopt(long)]
    vr: bool,

    #[structopt(flatten)]
    sub: mutautomata::Opt,
}

fn main() {
    let opt = Opt::from_args();
    let objects = mutautomata::generate(opt.sub);
    draw::draw(objects, opt.vr).expect("Draw failed");
}
