#version 450

// The `color_input` parameter of the `draw` method.
layout(input_attachment_index = 0, set = 0, binding = 1) uniform subpassInput u_diffuse;

layout(location = 0) out vec4 f_color;

void main() {
	vec3 color = vec3(0.1, 0.1, 0.1);

	// Load the value at the current pixel.
	vec3 in_diffuse = subpassLoad(u_diffuse).rgb;

	f_color.rgb = color.rgb * in_diffuse;
	f_color.a = 1.0;
}
