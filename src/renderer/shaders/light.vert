#version 450

// duplicated definition in model.frag too
layout(binding = 0) uniform CameraUniformBufferObject {
	mat4 view;
	mat4 proj;
	vec3 position;
} camera;

// in per vertex
layout(location = 0) in vec3 position;

// int per instance
// NOTE: mat4 takes 4 slots
layout(location = 4) in mat4 model; 
layout(location = 8) in vec3 material_ambient;
layout(location = 9) in vec3 material_diffuse;
layout(location = 10) in vec3 material_specular;
layout(location = 11) in float material_shininess;

// out
layout(location = 4) out vec3 f_material_ambient;
layout(location = 5) out vec3 f_material_diffuse;
layout(location = 6) out vec3 f_material_specular;
layout(location = 7) out float f_material_shininess;

void main() {
	gl_Position = camera.proj * camera.view * model * vec4(position, 1.0);

	f_material_ambient = material_ambient;
	f_material_diffuse = material_diffuse;
	f_material_specular = material_specular;
	f_material_shininess = material_shininess;
}
