#version 450

layout(location = 0) in vec2 texcoords;
layout(location = 1) in vec4 normals;

layout(binding = 2) uniform sampler2D texSampler;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(texSampler, texcoords);
}
