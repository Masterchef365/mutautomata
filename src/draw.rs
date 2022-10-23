use crate::DrawData;
use idek::{prelude::*, MultiPlatformCamera};

pub fn draw(objects: Vec<DrawData>, vr: bool) -> Result<()> {
    launch::<_, Mutautomata>(Settings {
        max_transforms: 10_000,
        msaa_samples: 4,
        vr,
        name: "Mutautomata".to_string(),
        args: objects,
    })
}

struct Mutautomata {
    camera: MultiPlatformCamera,
    cmds: Vec<DrawCmd>,
}

impl App<Vec<DrawData>> for Mutautomata {
    fn init(ctx: &mut Context, platform: &mut Platform, objects: Vec<DrawData>) -> Result<Self> {
        let points_shader = ctx.shader(
            &DEFAULT_VERTEX_SHADER,
            &DEFAULT_FRAGMENT_SHADER,
            Primitive::Points,
        )?;

        let lines_shader = ctx.shader(
            include_bytes!("shaders/unlit.vert.spv"),
            include_bytes!("shaders/unlit.frag.spv"),
            Primitive::Lines,
        )?;

        let cmds = objects
            .into_iter()
            .map(|obj| {
                Ok(DrawCmd::new(ctx.vertices(&obj.vertices, false)?)
                    .indices(ctx.indices(&obj.indices, false)?)
                    .shader(match obj.primitive {
                        Primitive::Points => points_shader,
                        Primitive::Lines => lines_shader,
                        _ => panic!(),
                    }))
            })
            .collect::<Result<Vec<DrawCmd>>>()?;

        Ok(Self {
            camera: MultiPlatformCamera::new(platform),
            cmds,
        })
    }

    fn frame(&mut self, _ctx: &mut Context, _: &mut Platform) -> Result<Vec<DrawCmd>> {
        Ok(self.cmds.clone())
    }

    fn event(
        &mut self,
        ctx: &mut Context,
        platform: &mut Platform,
        mut event: Event,
    ) -> Result<()> {
        if self.camera.handle_event(&mut event) {
            ctx.set_camera_prefix(self.camera.get_prefix())
        }
        idek::close_when_asked(platform, &event);
        Ok(())
    }
}
