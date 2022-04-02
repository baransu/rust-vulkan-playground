#version 450

struct Vertex {
    float vx, vy, vz;
    float nx, ny, nz, nw;
    float tx, ty;
};

layout(binding = 0) readonly buffer Vertices {
    Vertex vertices[];
};

layout(binding = 1) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) out vec2 texcoords;
layout(location = 1) out vec4 normals;

void main() {
    vec4 position = vec4(vertices[gl_VertexIndex].vx, vertices[gl_VertexIndex].vy, vertices[gl_VertexIndex].vz, 1.0);
    gl_Position = ubo.proj * ubo.view * ubo.model * position;

    texcoords = vec2(vertices[gl_VertexIndex].tx, vertices[gl_VertexIndex].ty);
    normals = vec4(vertices[gl_VertexIndex].nx, vertices[gl_VertexIndex].ny, vertices[gl_VertexIndex].nz, vertices[gl_VertexIndex].nw);
}
