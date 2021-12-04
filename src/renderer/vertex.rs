#[derive(Default, Copy, Clone)]
pub struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    color: [f32; 4],
}

vulkano::impl_vertex!(Vertex, position, normal, uv, color);

impl Vertex {
    pub fn new(position: [f32; 3], normal: [f32; 3], uv: [f32; 2], color: [f32; 4]) -> Vertex {
        Vertex {
            position,
            normal,
            uv,
            color,
        }
    }
}
