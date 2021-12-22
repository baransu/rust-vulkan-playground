#version 450

struct PointLight {
	vec3 position;
	vec3 color;
	// NOTE: is constant reserved keyword?
	float constant_;
	float linear;
	float quadratic;	
};


// duplicated definition in model.vert
layout(set = 0, binding = 0) uniform CameraUniformBufferObject {
	mat4 view;
	mat4 proj;
	vec3 position;
} camera;

layout(set = 1, binding = 0) uniform sampler2D diffuse_sampler;
layout(set = 1, binding = 1) uniform sampler2D normal_sampler;
layout(set = 1, binding = 2) uniform sampler2D metalic_roughness_sampler;

layout(set = 2, binding = 0) uniform samplerCube shadow_sampler;
layout(set = 2, binding = 1) uniform LightUniformBufferObject { 
	PointLight point_lights[32];
	int point_lights_count;
} lights;

// in
layout(location = 0) in vec2 f_uv;
layout(location = 1) in vec3 f_normal;
layout(location = 2) in vec4 f_tangent;
layout(location = 3) in vec4 f_position;
layout(location = 4) in vec3 f_position_world;

// out
layout(location = 0) out vec4 out_position;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec4 out_albedo;
layout(location = 3) out vec4 out_metalic_roughness;
layout(location = 4) out vec4 out_shadow_map;

vec3 gridSamplingDisk[20] = vec3[]
(
   vec3(1, 1,  1), vec3( 1, -1,  1), vec3(-1, -1,  1), vec3(-1, 1,  1), 
   vec3(1, 1, -1), vec3( 1, -1, -1), vec3(-1, -1, -1), vec3(-1, 1, -1),
   vec3(1, 1,  0), vec3( 1, -1,  0), vec3(-1, -1,  0), vec3(-1, 1,  0),
   vec3(1, 0,  1), vec3(-1,  0,  1), vec3( 1,  0, -1), vec3(-1, 0, -1),
   vec3(0, 1,  1), vec3( 0, -1,  1), vec3( 0, -1, -1), vec3( 0, 1, -1)
);

// TODO: move to light pass
const float farPlane = 100.0;
vec3 ShadowCalculation(vec3 fragPos, vec3 lightPos) {
	// get vector between fragment position and light position
	vec3 fragToLight = fragPos - lightPos;

	float currentDepth = length(fragToLight);

	float shadow = 0.0;
	float bias = 0.15;
	int samples = 20;

	float viewDistance = length(camera.position - fragPos);
	float diskRadius = (1.0 + (viewDistance / farPlane)) / 25.0;
	for(int i = 0; i < samples; ++i)
	{
			float closestDepth = texture(shadow_sampler, (fragToLight * vec3(1, -1, 1)) + gridSamplingDisk[i] * diskRadius).r;
			closestDepth *= farPlane;   // undo mapping [0;1]
			if(currentDepth - bias > closestDepth)
					shadow += 1.0;
	}
	shadow /= float(samples);

 	return vec3(shadow);
}

void main() {
	out_position = f_position;

	out_metalic_roughness = texture(metalic_roughness_sampler, f_uv);

	// normal
	vec3 tangentNormal = texture(normal_sampler, f_uv).xyz * 2.0 - 1.0;

	vec3 N  = normalize(f_normal);
	vec3 T  = normalize(f_tangent.xyz);
	vec3 B  = normalize(cross(N, T));
	mat3 TBN = mat3(T, B, N);

	out_normal = normalize(TBN * tangentNormal);

	vec3 shadow = vec3(0.0);

	for(int i = 0; i < lights.point_lights_count; i++) {
		shadow += ShadowCalculation(f_position_world.xyz, lights.point_lights[i].position); 
	}

	out_shadow_map = vec4(shadow, 1.0);

	out_albedo = texture(diffuse_sampler, f_uv);
	out_albedo.a = 1.0;
}
