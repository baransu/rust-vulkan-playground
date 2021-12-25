#version 450

struct PointLight {
	vec3 position;
	vec3 color;
	// NOTE: is constant reserved keyword?
	float constant_;
	float linear;
	float quadratic;	
};

struct DirLight {
	mat4 view;
	mat4 proj;
	vec3 direction;
};


// duplicated definition in model.frag too
layout(set = 0, binding = 0) uniform CameraUniformBufferObject {
	mat4 view;
	mat4 proj;
	vec3 position;
} camera;


layout(set = 2, binding = 2) uniform LightUniformBufferObject { 
	PointLight point_lights[32];
	DirLight dir_light;
	int point_lights_count;
} lights;

// in per vertex
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec4 tangent;

// int per instance
// NOTE: mat4 takes 4 slots
layout(location = 4) in mat4 model; 

// out
layout(location = 0) out vec2 f_uv;
layout(location = 1) out vec3 f_normal;
layout(location = 2) out vec4 f_tangent;
layout(location = 3) out vec4 f_position;
layout(location = 4) out vec4 f_position_light;

void main() {
	vec4 world_pos = model * vec4(position, 1.0);

	f_position_light = lights.dir_light.proj * lights.dir_light.view * world_pos;

	f_position = world_pos;

	gl_Position = camera.proj * camera.view * world_pos;

	f_uv = uv;

	f_tangent = vec4(mat3(model) * tangent.xyz, tangent.w);
	f_normal = mat3(model) * normal;
}
