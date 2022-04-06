pub mod gbuffer_shader {
    vulkano_shaders::shader! {
        shaders: {
            vertex: {
                ty: "vertex",
                path: "src/renderer/shaders/gbuffer.vert",
            },
            fragment: {
                ty: "fragment",
                path: "src/renderer/shaders/gbuffer.frag"
            },
        },
        types_meta: {
            #[derive(Clone, Copy, Default)]
        }
    }
}

pub type Vertex = gbuffer_shader::ty::Vertex;
pub type CameraUniformBufferObject = gbuffer_shader::ty::CameraUniformBufferObject;
pub type LightSpaceUniformBufferObject = shadow_vertex_shader::ty::LightSpaceUniformBufferObject;

pub mod shadow_vertex_shader {
    vulkano_shaders::shader! {
                                                                    ty: "vertex",
                                                                    src: "
			#version 450

			layout(binding = 0) uniform LightSpaceUniformBufferObject {
				mat4 matrix;
			} light_space;

			// per vertex
			layout(location = 1) in vec3 position;

			// per instance
			layout(location = 4) in mat4 model; 

			void main() {
				gl_Position = light_space.matrix * model * vec4(position, 1.0);
			}									
"
    }
}

pub mod shadow_fs {
    vulkano_shaders::shader! {
                                    ty: "fragment",
                                    src: "
			#version 450

			void main() {
			}
"
    }
}

pub mod screen_vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/renderer/shaders/fullscreen.vert"
    }
}

pub mod screen_fragment_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/renderer/shaders/screen.frag"
    }
}
