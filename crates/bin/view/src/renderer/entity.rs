use super::transform::Transform;

pub struct Entity {
    pub model_id: String,

    pub transform: Transform,
}

impl Entity {
    pub fn new(model_id: &str, transform: Transform) -> Entity {
        Entity {
            model_id: model_id.to_string(),
            transform,
        }
    }
}

#[derive(Default, Copy, Clone)]
pub struct InstanceData {
    pub model: [[f32; 4]; 4],
}

vulkano::impl_vertex!(InstanceData, model);
