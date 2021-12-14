#version 450

layout(binding = 0) uniform sampler2D ssao;

// in
layout(location = 0) in vec2 f_uv;

// out
layout(location = 0) out float out_color;

void main() {
	out_color = texture(ssao, f_uv).r;

	vec2 texel_size = 1.0 / vec2(textureSize(ssao, 0));
	float result = 0.0;
	
	for (int x = -2; x < 2; ++x) 
	{
			for (int y = -2; y < 2; ++y) 
			{
					vec2 offset = vec2(float(x), float(y)) * texel_size;
					result += texture(ssao, f_uv + offset).r;
			}
	}

	out_color = result / (4.0 * 4.0);
}
