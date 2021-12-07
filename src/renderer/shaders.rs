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

pub mod screen_vertex_shader {
    vulkano_shaders::shader! {
                                    ty: "vertex",
                                    src: "
				#version 450

				layout(location = 0) in vec2 position;
				layout(location = 2) in vec2 uv;
	
				layout(location = 0) out vec2 outUV;
				
				void main() {
					outUV = uv;
					gl_Position = vec4(position, 0.0, 1.0);
				}									
	"
    }
}

pub mod screen_fragment_shader {
    vulkano_shaders::shader! {
                                    ty: "fragment",
                                    src: "
		#version 450

		layout (set = 0, binding = 0) uniform sampler2D screen_texture;
		
		layout (location = 0) in vec2 inUV;
		
		layout (location = 0) out vec4 outFragColor;

		vec4 no_effect() {
			return texture(screen_texture, inUV);	
		}

		vec4 negative_effect() {
			return vec4(vec3(1.0 - texture(screen_texture, inUV)), 1.0);
		}

		vec4 grayscale_effect() {
			vec4 texture_color = texture(screen_texture, inUV);
			float average = 0.2126 * texture_color.r + 0.7152 * texture_color.g + 0.0722 * texture_color.b;
			return vec4(average, average, average, 1.0);
		}

		vec4 kernel_effect(float kernel[9]) {
			const float offset = 1.0 / 300.0; 
			vec2 offsets[9] = vec2[] (
				vec2(-offset,  offset), // top-left
				vec2( 0.0f,    offset), // top-center
				vec2( offset,  offset), // top-right
				vec2(-offset,  0.0f),   // center-left
				vec2( 0.0f,    0.0f),   // center-center
				vec2( offset,  0.0f),   // center-right
				vec2(-offset, -offset), // bottom-left
				vec2( 0.0f,   -offset), // bottom-center
				vec2( offset, -offset)  // bottom-right    
			);
			
			vec3 sampleTex[9];
			for(int i = 0; i < 9; i++)
					sampleTex[i] = vec3(texture(screen_texture, inUV.xy + offsets[i]));

			vec3 col = vec3(0.0);
			for(int i = 0; i < 9; i++)
					col += sampleTex[i] * kernel[i];
			
			return vec4(col, 1.0);
		}

		vec4 sharpen_effect() {
			float kernel[9] = float[](
				-1, -1, -1,
				-1,  9, -1,
				-1, -1, -1
			);

			return kernel_effect(kernel);
		}

		vec4 blur_effect() {
			float kernel[9] = float[](
				1.0 / 16, 2.0 / 16, 1.0 / 16,
				2.0 / 16, 4.0 / 16, 2.0 / 16,
				1.0 / 16, 2.0 / 16, 1.0 / 16  
			);

			return kernel_effect(kernel);
		}

		vec4 edge_detection_effect() {
			float kernel[9] = float[](
				1,  1, 1,
				1, -8, 1,
				1,  1, 1
			);

			return kernel_effect(kernel);
		}

		void main() {
			outFragColor = no_effect();	
			// outFragColor = negative_effect();
			// outFragColor = grayscale_effect();
			// outFragColor = sharpen_effect();
			// outFragColor = blur_effect();
			// outFragColor = edge_detection_effect();
		}
	"
    }
}

pub mod skybox_vertex_shader {
    vulkano_shaders::shader! {
                                                                    ty: "vertex",
                                                                    src: "
			#version 450

			layout(location = 0) in vec3 position;

			layout(binding = 0) uniform MVPUniformBufferObject {
				mat4 view;
				mat4 proj;
				mat4 model;
		 } mvp_ubo;

			layout(location = 0) out vec3 outUV;
			
			void main() {
				outUV = position;
				gl_Position = mvp_ubo.proj * mvp_ubo.view * vec4(position, 1.0);
			}									
"
    }
}

pub mod skybox_fragment_shader {
    vulkano_shaders::shader! {
                    ty: "fragment",
                    src: "
	#version 450

	layout (binding = 1) uniform samplerCube skybox_texture;
	
	layout (location = 0) in vec3 inUV;
	
	layout (location = 0) out vec4 outFragColor;

	void main() {
		vec3 uv = vec3(inUV.x, -inUV.y, inUV.z);
		outFragColor = texture(skybox_texture, uv);
	}
"
    }
}
