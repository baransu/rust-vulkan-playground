#version 450

// duplicated definition in model.vert
layout(set = 0, binding = 0) uniform CameraUniformBufferObject {
	mat4 view;
	mat4 proj;
	vec3 position;
} camera;

layout(set = 1, binding = 0) uniform sampler2D diffuse_sampler;
layout(set = 1, binding = 1) uniform sampler2D normal_sampler;
layout(set = 1, binding = 2) uniform sampler2D metalic_roughness_sampler;

// in
layout(location = 0) in vec2 f_uv;
layout(location = 1) in vec3 f_normal;
layout(location = 2) in vec4 f_tangent;
layout(location = 3) in vec4 f_position;

// out
layout(location = 0) out vec4 out_position;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec4 out_albedo;
layout(location = 3) out vec4 out_metalic_roughness;

void main() {
	out_position = f_position;

	out_albedo = texture(diffuse_sampler, f_uv);
	out_albedo.a = 1.0;

	out_metalic_roughness = texture(metalic_roughness_sampler, f_uv);

	// normal
	vec3 tangentNormal = texture(normal_sampler, f_uv).xyz * 2.0 - 1.0;

	vec3 N  = normalize(f_normal);
	vec3 T  = normalize(f_tangent.xyz);
	vec3 B  = normalize(cross(N, T));
	mat3 TBN = mat3(T, B, N);

	out_normal = normalize(TBN * tangentNormal);
}
