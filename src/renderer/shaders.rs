pub mod model_vertex_shader {
    vulkano_shaders::shader! {
                ty: "vertex",
                path: "src/renderer/shaders/gbuffer.vert"
    }
}

pub mod model_fragment_shader {
    vulkano_shaders::shader! {
                ty: "fragment",
                path: "src/renderer/shaders/gbuffer.frag"
    }
}

pub type CameraUniformBufferObject = model_vertex_shader::ty::CameraUniformBufferObject;
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
                                    src: "
		#version 450

		layout (set = 0, binding = 0) uniform sampler2D screen_texture;
		layout (set = 0, binding = 1) uniform sampler2D ui_texture;
		layout (set = 0, binding = 2) uniform sampler2D shadow_texture;
		
		layout (location = 0) in vec2 inUV;
		
		layout (location = 0) out vec4 outFragColor;

		vec4 no_effect() {
			// for shadow map testing
			// float depthValue = texture(shadow_texture, inUV).r;
			// return vec4(vec3(depthValue), 1.0);
	
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
			// vec4 color = negative_effect();
			// vec4 color = grayscale_effect();
			// vec4 color = sharpen_effect();
			// vec4 color = blur_effect();
			// vec4 color = edge_detection_effect();
			vec4 color = no_effect();

			vec4 ui = texture(ui_texture, inUV);
			outFragColor = mix(color, ui, ui.a);
		}
	"
    }
}
