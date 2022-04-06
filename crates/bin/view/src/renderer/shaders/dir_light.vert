#version 450

struct Vertex {
	float px, py, pz;
	float nx, ny, nz;
  float ux, uy;
  float tx, ty, tz, tw;
};

layout(set = 0, binding = 0) readonly buffer Vertices {
	Vertex vertices[];
};

layout(set = 1, binding = 0) uniform CameraUniformBufferObject {
	mat4 view;
	mat4 proj;
	vec3 position;
} camera;

// per instance
layout(location = 0) in mat4 model; 

void main() {
	vec4 position = vec4(vertices[gl_VertexIndex].px, vertices[gl_VertexIndex].py, vertices[gl_VertexIndex].pz, 1.0);
	gl_Position = camera.proj * camera.view * model * position;
}				
