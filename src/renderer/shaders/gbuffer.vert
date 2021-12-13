#version 450

// duplicated definition in model.frag too
layout(binding = 0) uniform CameraUniformBufferObject {
	mat4 view;
	mat4 proj;
	vec3 position;
} camera;

// in per vertex
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

// int per instance
// NOTE: mat4 takes 4 slots
layout(location = 3) in mat4 model; 
layout(location = 7) in vec3 material_diffuse;
layout(location = 8) in vec3 material_specular;

// out
layout(location = 0) out vec2 f_uv;
layout(location = 1) out vec3 f_normal;
layout(location = 2) out vec3 f_position;
layout(location = 3) out vec3 f_material_diffuse;
layout(location = 4) out vec3 f_material_specular;

void main() {
	f_position = vec3(model * vec4(position, 1.0));

	gl_Position = camera.proj * camera.view * vec4(f_position, 1.0);

	f_uv = uv;

	f_material_diffuse = material_diffuse;
	f_material_specular = material_specular;

	f_normal = mat3(transpose(inverse(model))) * normal;
}
