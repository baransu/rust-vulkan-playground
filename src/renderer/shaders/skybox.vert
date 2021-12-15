#version 450

layout(location = 0) in vec3 position;

layout(binding = 0) uniform CameraUniformBufferObject {
	mat4 view;
	mat4 proj;
	mat4 position;
} camera;

layout(location = 0) out vec3 WorldPos;

void main() {
	WorldPos = position;
	gl_Position = camera.proj * camera.view * vec4(position, 1.0);
}	
