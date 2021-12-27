#version 450

layout (location = 0) in vec3 position;

layout (location = 0) out vec3 localPos;

layout (binding = 0) uniform UBO {
		mat4 view;
		mat4 projection;
} ubo;

void main() {
	localPos = position;  
	gl_Position = ubo.projection * ubo.view * vec4(position, 1.0);
}
