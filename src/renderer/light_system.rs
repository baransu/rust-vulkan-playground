use std::sync::Arc;

use vulkano::{
    pipeline::{vertex::BuffersDefinition, GraphicsPipeline},
    render_pass::Subpass,
};

use super::{context::Context, mesh::InstanceData, vertex::Vertex};

pub struct LightSystem {
    pub pipeline: Arc<GraphicsPipeline>,
}

impl LightSystem {
    pub fn initialize(context: &Context, subpass: Subpass) -> LightSystem {
        let pipeline = Self::create_pipeline(context, subpass);

        LightSystem { pipeline }
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
                .render_pass(subpass)
                .build(context.device.clone())
                .unwrap(),
        )
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/renderer/shaders/light.vert"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/renderer/shaders/light.frag"
    }
}

pub type LightUniformBufferObject = fs::ty::LightUniformBufferObject;
pub type ShaderPointLight = fs::ty::PointLight;
pub type ShaderDirectionalLight = fs::ty::DirectionalLight;
