pub mod vs {
    vulkano_shaders::shader! {
            ty: "vertex",
            src: "
			#version 450

			layout(push_constant) uniform VertPC {
				mat4 matrix;
			};
			
			layout(location = 0) in vec2 pos;
			layout(location = 1) in vec2 uv;
			layout(location = 2) in uint col;
			
			layout(location = 0) out vec2 f_uv;
			layout(location = 1) out vec4 f_color;
			
			void main() {
					f_uv = uv;
					f_color = unpackUnorm4x8(col);
					gl_Position = matrix * vec4(pos.xy, 0, 1);
			}
						",
    }
}

pub type TextureUsage = fs::ty::TextureUsage;

pub mod fs {
    vulkano_shaders::shader! {
            ty: "fragment",
            src: "
			#version 450

			layout(binding = 0) uniform sampler2D tex;
			layout(binding = 1) uniform TextureUsage {
				bool depth;
			} usage;

			layout(location = 0) in vec2 f_uv;
			layout(location = 1) in vec4 f_color;

			layout(location = 0) out vec4 Target0;

			void main() {
				if(usage.depth) {
					float depth = texture(tex, f_uv).r;
					Target0 = vec4(vec3(depth), 1.0);
				} else {
					Target0 = f_color * texture(tex, f_uv);
				}
			
			}		
						",
    }
}
