#version 450

layout(location = 0) in vec3 position;

layout(binding = 0) uniform SceneUniformBufferObject {
	mat4 view;
	mat4 proj;
} scene;

layout(location = 0) out vec3 WorldPos;

void main() {
	WorldPos = position;
	gl_Position = scene.proj * scene.view * vec4(position, 1.0);
}	
