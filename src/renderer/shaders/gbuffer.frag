#version 450

// duplicated definition in model.vert
layout(binding = 0) uniform CameraUniformBufferObject {
	mat4 view;
	mat4 proj;
	vec3 position;
} camera;

layout(binding = 1) uniform sampler2D diffuse_sampler;
layout(binding = 2) uniform sampler2D normal_sampler;
layout(binding = 3) uniform sampler2D metalic_roughness_sampler;

// in
layout(location = 0) in vec2 f_uv;
layout(location = 1) in vec3 f_normal;
layout(location = 2) in vec3 f_tangent;
layout(location = 3) in vec3 f_position;
layout(location = 4) in vec3 f_material_diffuse;
layout(location = 5) in vec3 f_material_specular;

// out
layout(location = 0) out vec4 out_position;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec4 out_albedo;
layout(location = 3) out vec4 out_metalic_roughness;

float linear_depth(float depth) {
	float z = depth * 2.0 - 1.0; 
	return (2.0 * 0.1 * 1000.0) / (0.1 + 1000.0 - z * (1000.0 - 0.1));	
}

void main() {
	out_position = vec4(f_position, linear_depth(gl_FragCoord.z));

	out_albedo.rgb = f_material_diffuse.rgb * texture(diffuse_sampler, f_uv).rgb;
	out_albedo.a = f_material_specular.r;

	out_metalic_roughness = texture(metalic_roughness_sampler, f_uv);

	// normal
	vec3 tangentNormal = texture(normal_sampler, f_uv).xyz * 2.0 - 1.0;

	vec3 Q1  = dFdx(f_position);
	vec3 Q2  = dFdy(f_position);
	vec2 st1 = dFdx(f_uv);
	vec2 st2 = dFdy(f_uv);

	vec3 N   = normalize(f_normal);
	vec3 T  = normalize(Q1*st2.t - Q2*st1.t);
	vec3 B  = -normalize(cross(N, T));
	mat3 TBN = mat3(T, B, N);

	out_normal = normalize(TBN * tangentNormal);
}
