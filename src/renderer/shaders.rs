pub mod model_vertex_shader {
    vulkano_shaders::shader! {
                    ty: "vertex",
                    src: "
				#version 450

				layout(binding = 0) uniform MVPUniformBufferObject {
						mat4 view;
						mat4 proj;
						mat4 model;
				} mvp_ubo;

				layout(location = 0) in vec3 position;
				layout(location = 1) in vec3 normal;
				layout(location = 2) in vec2 uv;
				layout(location = 3) in vec4 color;

				layout(location = 0) out vec2 f_uv;
				layout(location = 1) out vec3 f_normal;
				layout(location = 2) out vec3 f_position;

				out gl_PerVertex {
						vec4 gl_Position;
				};

				void main() {
						gl_Position = mvp_ubo.proj * mvp_ubo.view * mvp_ubo.model * vec4(position, 1.0);
						f_position = vec3(mvp_ubo.model * vec4(position, 1.0));
						f_uv = uv;
						f_normal = mat3(transpose(inverse(mvp_ubo.model))) * normal;  
				}
		"
    }
}

pub mod model_fragment_shader {
    vulkano_shaders::shader! {
                    ty: "fragment",
                    src: "
				#version 450

				layout(binding = 1) uniform sampler2D tex_sampler;

				layout(location = 0) in vec2 f_uv;
				layout(location = 1) in vec3 f_normal;
				layout(location = 2) in vec3 f_position;

				layout(location = 0) out vec4 out_color;

				void main() {
						vec3 light_pos = vec3(5.0, 0.0, 0.0);
						vec3 light_color = vec3(1.0, 1.0, 1.0);
						vec3 ambient = 0.1 * light_color;

						vec3 norm = normalize(f_normal);
						vec3 light_dir = normalize(light_pos - f_position);  

						float diff = max(dot(norm, light_dir), 0.0);
						vec3 diffuse = diff * light_color;
						vec3 result = (ambient + diffuse) * texture(tex_sampler, f_uv).xyz;
						out_color = vec4(result,  1.0);
				}
		"
    }
}

pub type MVPUniformBufferObject = model_vertex_shader::ty::MVPUniformBufferObject;

pub mod blur_vertex_shader {
    vulkano_shaders::shader! {
                                    ty: "vertex",
                                    src: "
				#version 450

				layout(location = 0) in vec3 position;
				layout(location = 1) in vec3 normal;
				layout(location = 2) in vec2 uv;
				layout(location = 3) in vec4 color;

				layout(location = 0) out vec2 outUV;
				
				void main() {
					outUV = uv;
					gl_Position = vec4(position, 1.0);
				}									
	"
    }
}

pub mod blur_fragment_shader {
    vulkano_shaders::shader! {
                                    ty: "fragment",
                                    src: "
		#version 450

		layout (set = 0, binding = 0) uniform sampler2D screen_texture;
		
		layout (location = 0) in vec2 inUV;
		
		layout (location = 0) out vec4 outFragColor;
		
		void main() {
			outFragColor = texture(screen_texture, inUV);
		}
	"
    }
}
