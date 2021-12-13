#version 450

// duplicated definition in model.vert
layout(binding = 0) uniform CameraUniformBufferObject {
	mat4 view;
	mat4 proj;
	vec3 position;
} camera;

layout(binding = 1) uniform sampler2D diffuse_sampler;
layout(binding = 2) uniform sampler2D normal_sampler;

// in
layout(location = 0) in vec2 f_uv;
layout(location = 1) in vec3 f_normal;
layout(location = 2) in vec3 f_position;
layout(location = 3) in vec3 f_material_diffuse;
layout(location = 4) in vec3 f_material_specular;

// out
layout(location = 0) out vec3 out_position;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec4 out_albedo;

void main() {
	out_position = f_position;
	out_normal = normalize(f_normal);

	out_albedo.rgb = f_material_diffuse * texture(diffuse_sampler, f_uv).rgb;
	out_albedo.a = f_material_specular.r;
}
