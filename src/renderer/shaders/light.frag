#version 450

#define MAX_POINT_LIGHTS 32

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

// The `color_input` parameter of the `draw` method.
layout(binding = 1) uniform sampler2D u_position;
layout(binding = 2) uniform sampler2D u_normals;
layout(binding = 3) uniform sampler2D u_albedo;
layout(binding = 4) uniform sampler2D u_metalic_roughness;
layout(binding = 5) uniform sampler2D ssao_sampler;
layout(binding = 6) uniform sampler2D shadow_sampler;

// duplicated definition in model.vert and shaders.rs
layout(binding = 7)	uniform LightSpaceUniformBufferObject {
	mat4 matrix;
} light_space;

layout(binding = 8) uniform LightUniformBufferObject { 
	PointLight point_lights[MAX_POINT_LIGHTS];
	DirectionalLight dir_light;
	int point_lights_count;
} light;

layout(location = 0) in vec2 f_uv;

layout(location = 0) out vec4 out_color;

float shadow_calculation_pcf(vec4 f_position_light_space, vec3 normal, vec3 light_dir) {
	// perform perspective divide
	vec3 proj_coords = f_position_light_space.xyz / f_position_light_space.w;

	// transform to [0,1] range
	proj_coords = proj_coords * 0.5 + 0.5;

	// get depth of current fragment from light's perspective
	float current_depth = proj_coords.z;

	float bias = max(0.05 * (1.0 - dot(normal, light_dir)), 0.005);  

	float shadow = 0.0;
	vec2 texel_size = 1.0 / textureSize(shadow_sampler, 0);
	for(int x = -1; x <= 1; ++x)
	{
			for(int y = -1; y <= 1; ++y)
			{
					float pcf_depth = texture(shadow_sampler, proj_coords.xy + vec2(x, y) * texel_size).r; 
					shadow += current_depth - bias > pcf_depth ? 1.0 : 0.0;        
			}    
	}

	shadow /= 9.0;

	if(proj_coords.z > 1.0) {
		shadow = 0.0;
	}

	return shadow;
}

vec3 calc_dir_light(DirectionalLight light, vec3 f_normal, vec3 f_position, float f_specular, vec3 diffuse, vec4 f_position_light_space) {
	vec3 normal = f_normal;
	vec3 position = vec4(f_position, 1.0).xyz;
	
	vec3 view_dir = normalize(-position);

	vec3 light_dir = normalize(-light.direction);
	
	// diffuse shading
	vec3 diff = max(dot(normal, light_dir), 0.0) * light.diffuse * diffuse;
	
	// specular shading
	vec3 halfway_dir = normalize(light_dir + view_dir);
	vec3 spec = pow(max(dot(normal, halfway_dir), 0.0), 32) * light.specular * f_specular;

	// shadows
	float shadow = shadow_calculation_pcf(f_position_light_space, normal, light_dir);       

	return (1.0 - shadow) * (diff + spec);   
}

vec3 calc_point_light(PointLight light, vec3 f_normal, vec3 f_position, float f_specular, vec3 diffuse) {
	vec3 normal = f_normal;
	vec3 position = vec4(f_position, 1.0).xyz;

	vec3 view_dir = normalize(-position);

	vec3 light_dir = normalize(light.position - position);

	// diffuse shading
	vec3 diff = max(dot(normal, light_dir), 0.0) * light.diffuse * diffuse;

	// specular shading
	vec3 halfway_dir = normalize(light_dir + view_dir);
	vec3 specular = pow(max(dot(normal, halfway_dir), 0.0), 32) * light.specular * f_specular;

	// attenuation
	float dist = length(light.position - position);
	float attenuation = 1.0 / (light.constant_ + light.linear * dist + light.quadratic * dist * dist);    

	diff *= attenuation;
	specular *= attenuation;

	return diff + specular;   
}

void main() {
	vec3 f_position = texture(u_position, f_uv).xyz;
	vec3 f_normal = texture(u_normals, f_uv).rgb;
	vec4 f_position_light_space = light_space.matrix * vec4(f_position, 1.0);

	float ambient_occlusion = texture(ssao_sampler, f_uv).r;
	vec3 diffuse = texture(u_albedo, f_uv).rgb;
	float f_specular = texture(u_albedo, f_uv).a;

	// phase 1: ambient occlusion
	vec3 ambient = diffuse * 0.3 * ambient_occlusion;

	vec3 result = ambient;

	// phase 2: directional light with shadows
	result += calc_dir_light(light.dir_light, f_normal, f_position, f_specular, diffuse, f_position_light_space);

	// phase 3: point lights
	for(int i = 0; i < light.point_lights_count; i++)
    result += calc_point_light(light.point_lights[i], f_normal, f_position, f_specular, diffuse);

	// phase 4: texture
	vec4 color = vec4(result, 1.0);

	// phase 5: exposure tone mapping
	float exposure = 1.0;
	vec3 mapped = vec3(1.0) - exp(-color.rgb * exposure);

	// phase 6: gamma correction
	float gamma = 2.2;
	mapped = pow(mapped, vec3(1.0 / gamma));

	out_color = vec4(mapped, 1.0);
}
