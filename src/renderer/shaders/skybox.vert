#version 450

layout(binding = 0) uniform CameraUniformBufferObject {
	mat4 view;
	mat4 proj;
	mat4 position;
} camera;

layout(location = 0) in vec3 position;

layout(location = 0) out vec3 WorldPos;

void main() {
	// NOTE: skybox cube is flipped
	WorldPos = position * vec3(1.0, -1.0, 1.0);
	gl_Position = camera.proj * camera.view * vec4(position, 1.0);
}	
