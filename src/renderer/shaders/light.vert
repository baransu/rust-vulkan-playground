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

void main() {
	gl_Position = camera.proj * camera.view * model * vec4(position, 1.0);
}
