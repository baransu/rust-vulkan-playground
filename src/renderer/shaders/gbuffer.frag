#version 450

#define NR_POINT_LIGHTS 4

struct DirectionalLight { 
	vec3 direction;
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;
};

struct PointLight {
	vec3 position;
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;
	// NOTE: is constant reserved keyword?
	float constant_;
	float linear;
	float quadratic;	
};

// duplicated definition in model.vert
layout(binding = 0) uniform CameraUniformBufferObject {
	mat4 view;
	mat4 proj;
	vec3 position;
} camera;

layout(binding = 1) uniform sampler2D diffuse_sampler;
layout(binding = 2) uniform sampler2D shadow_sampler;

// duplicated definition in model.vert and shaders.rs
layout(binding = 3) uniform LightSpaceUniformBufferObject {
	mat4 matrix;
} light_space;

layout(binding = 4) uniform LightUniformBufferObject { 
	PointLight point_lights[NR_POINT_LIGHTS];
	DirectionalLight dir_light;
} light;

// in
layout(location = 0) in vec2 f_uv;
layout(location = 1) in vec3 f_normal;
layout(location = 2) in vec3 f_position;
layout(location = 3) in vec4 f_color;
layout(location = 4) in vec3 f_material_ambient;
layout(location = 5) in vec3 f_material_diffuse;
layout(location = 6) in vec3 f_material_specular;
layout(location = 7) in float f_material_shininess;
layout(location = 8) in vec4 f_position_light_space;

// out
layout(location = 0) out vec4 out_color;
layout(location = 1) out vec3 out_normal;


void main() {
	out_color = texture(diffuse_sampler, f_uv);

	out_normal = f_normal;
}
