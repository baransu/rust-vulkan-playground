use std::sync::Arc;

use vulkano::{
    pipeline::{
        blend::{AttachmentBlend, BlendFactor, BlendOp},
        vertex::BuffersDefinition,
        GraphicsPipeline,
    },
    render_pass::Subpass,
};

use super::{context::Context, mesh::InstanceData, vertex::Vertex};

pub struct AmbientLightSystem {
    pub pipeline: Arc<GraphicsPipeline>,
}

impl AmbientLightSystem {
    pub fn initialize(context: &Context, subpass: Subpass) -> AmbientLightSystem {
        let pipeline = Self::create_pipeline(context, subpass);

        AmbientLightSystem { pipeline }
    }

    fn create_pipeline(context: &Context, subpass: Subpass) -> Arc<GraphicsPipeline> {
        let vs = vs::Shader::load(context.graphics_queue.device().clone()).unwrap();
        let fs = fs::Shader::load(context.graphics_queue.device().clone()).unwrap();

        Arc::new(
            GraphicsPipeline::start()
                .vertex_input(
                    BuffersDefinition::new()
                        .vertex::<Vertex>()
                        .instance::<InstanceData>(),
                )
                .vertex_shader(vs.main_entry_point(), ())
                .triangle_list()
                .viewports_dynamic_scissors_irrelevant(1)
                .fragment_shader(fs.main_entry_point(), ())
                .blend_collective(AttachmentBlend {
                    enabled: true,
                    color_op: BlendOp::Add,
                    color_source: BlendFactor::One,
                    color_destination: BlendFactor::One,
                    alpha_op: BlendOp::Max,
                    alpha_source: BlendFactor::One,
                    alpha_destination: BlendFactor::One,
                    mask_red: true,
                    mask_green: true,
                    mask_blue: true,
                    mask_alpha: true,
                })
                .render_pass(subpass)
                .build(context.device.clone())
                .unwrap(),
        )
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/renderer/shaders/ambient_light.vert"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/renderer/shaders/ambient_light.frag"
    }
}
