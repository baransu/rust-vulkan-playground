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

// duplicated definition in model.vert
layout(set = 0, binding = 0) uniform CameraUniformBufferObject {
	mat4 view;
	mat4 proj;
	vec3 position;
} camera;

layout(set = 1, binding = 0) uniform sampler2D diffuse_sampler;
layout(set = 1, binding = 1) uniform sampler2D normal_sampler;
layout(set = 1, binding = 2) uniform sampler2D metalic_roughness_sampler;

layout(set = 2, binding = 0) uniform samplerCube point_shadow_sampler;
layout(set = 2, binding = 1) uniform sampler2D dir_shadow_sampler;
layout(set = 2, binding = 2) uniform LightUniformBufferObject { 
	PointLight point_lights[32];
	DirLight dir_light;
	int point_lights_count;
} lights;

// in
layout(location = 0) in vec2 f_uv;
layout(location = 1) in vec3 f_normal;
layout(location = 2) in vec4 f_tangent;
layout(location = 3) in vec4 f_position;
layout(location = 4) in vec3 f_position_world;
layout(location = 5) in vec4 f_position_light;

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
vec3 PointShadowCalculation(vec3 fragPos, vec3 lightPos) {
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
			float closestDepth = texture(point_shadow_sampler, (fragToLight * vec3(1, -1, 1)) + gridSamplingDisk[i] * diskRadius).r;
			closestDepth *= farPlane;   // undo mapping [0;1]
			if(currentDepth - bias > closestDepth)
					shadow += 1.0;
	}
	shadow /= float(samples);

 	return vec3(shadow);
}


float DirShadowCalculation(vec4 fragPosLightSpace, vec3 normal)
{
    // perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
   
    // get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;

		// float closestDepth = texture(dir_shadow_sampler, projCoords.xy).r;

		float bias = 0.15; // max(0.05 * (1.0 - dot(normal, lights.dir_light.direction)), 0.005); 

		float shadow = 0.0;
		vec2 texelSize = 1.0 / textureSize(dir_shadow_sampler, 0);
		for(int x = -1; x <= 1; ++x)
		{
				for(int y = -1; y <= 1; ++y)
				{
						float pcfDepth = texture(dir_shadow_sampler, projCoords.xy + vec2(x, y) * texelSize).r; 
						shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;        
				}    
		}
		shadow /= 9.0;

		if(projCoords.z > 1.0) {
        shadow = 0.0;
		}

		return shadow;
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
		// shadow += PointShadowCalculation(f_position_world.xyz, lights.point_lights[i].position); 
	}

	shadow += DirShadowCalculation(f_position_light, out_normal);

	out_shadow_map = vec4(shadow, 1.0);

	out_albedo = texture(diffuse_sampler, f_uv);
	out_albedo.a = 1.0;
}
