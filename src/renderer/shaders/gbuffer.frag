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
layout(location = 2) in vec3 f_tangent;
layout(location = 3) in vec3 f_position;
layout(location = 4) in vec3 f_material_diffuse;
layout(location = 5) in vec3 f_material_specular;

// out
layout(location = 0) out vec4 out_position;
layout(location = 1) out vec4 out_normal;
layout(location = 2) out vec4 out_albedo;

float linear_depth(float depth) {
	float z = depth * 2.0 - 1.0; 
	return (2.0 * 0.1 * 1000.0) / (0.1 + 1000.0 - z * (1000.0 - 0.1));	
}

void main() {
	out_position = vec4(f_position, linear_depth(gl_FragCoord.z));

	vec3 t = normalize(f_tangent);
	vec3 b = normalize(cross(f_normal, f_tangent));
	vec3 n = cross(t, b);

	vec3 normal = texture(normal_sampler, f_uv).rgb;
	normal = normalize(normal * 2.0 - 1.0);
	normal = normal.x * t + normal.y * b + normal.z * n;
	out_normal = vec4(normal, 1.0);

	out_albedo.rgb = f_material_diffuse.rgb * texture(diffuse_sampler, f_uv).rgb;
	out_albedo.a = f_material_specular.r;
}
