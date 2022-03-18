#version 450

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec2 f_uv;

void main() {
	f_uv = vec2(uv.x, 1.0 - uv.y);
	gl_Position = vec4(position, 1.0, 1.0);
}									
