#version 450

struct PointLight {
	vec3 position;
	vec3 color;
	// NOTE: is constant reserved keyword?
	float constant_;
	float linear;
	float quadratic;	
};

struct Vertex {
	float px, py, pz;
	float nx, ny, nz;
  float ux, uy;
  float tx, ty, tz, tw;
};

layout(set = 0, binding = 0) readonly buffer Vertices {
	Vertex vertices[];
};

// duplicated definition in model.frag too
layout(set = 1, binding = 0) uniform CameraUniformBufferObject {
	mat4 view;
	mat4 proj;
	vec3 position;
} camera;

// int per instance
// NOTE: mat4 takes 4 slots
layout(location = 0) in mat4 model; 

// out
layout(location = 0) out vec2 f_uv;
layout(location = 1) out vec3 f_normal;
layout(location = 2) out vec4 f_tangent;
layout(location = 3) out vec4 f_position;

void main() {
	vec3 position = vec3(vertices[gl_VertexIndex].px, vertices[gl_VertexIndex].py, vertices[gl_VertexIndex].pz);
	vec2 uv = vec2(vertices[gl_VertexIndex].ux, vertices[gl_VertexIndex].uy);
	vec3 normal = vec3(model * vec4(vertices[gl_VertexIndex].nx, vertices[gl_VertexIndex].ny, vertices[gl_VertexIndex].nz, 0.0));
	vec4 tangent = vec4(vertices[gl_VertexIndex].tx, vertices[gl_VertexIndex].ty, vertices[gl_VertexIndex].tz, vertices[gl_VertexIndex].tw);

	vec4 world_pos = model * vec4(position, 1.0);

	f_position = world_pos;

	gl_Position = camera.proj * camera.view * world_pos;

	f_uv = uv;

	f_tangent = vec4(mat3(model) * tangent.xyz, tangent.w);
	f_normal = mat3(model) * normal;
}
