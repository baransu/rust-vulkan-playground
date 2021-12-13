#[derive(Default, Copy, Clone)]
pub struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    tangent: [f32; 4],
}

vulkano::impl_vertex!(Vertex, position, normal, uv, tangent);

impl Vertex {
    pub fn new(position: [f32; 3], normal: [f32; 3], uv: [f32; 2], tangent: [f32; 4]) -> Vertex {
        Vertex {
            position,
            normal,
            uv,
            tangent,
        }
    }
}
