#version 450

#define NR_POINT_LIGHTS 1

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

vec3 calc_dir_light(DirectionalLight light, vec3 normal, vec3 view_dir) {
	vec3 light_dir = normalize(-light.direction);
	
	// diffuse shading
	float diff = max(dot(normal, light_dir), 0.0);
	
	// specular shading
	vec3 reflect_dir = reflect(-light_dir, normal);
	float spec = pow(max(dot(view_dir, reflect_dir), 0.0), f_material_shininess);

	// combine results
	vec3 ambient  = light.ambient * f_material_ambient;
	vec3 diffuse  = light.diffuse * diff * f_material_diffuse;
	vec3 specular = light.specular * spec * f_material_specular;
	return ambient + diffuse + specular;
}

float shadow_calculation(vec4 f_position_light_space) {
	// perform perspective divide
	vec3 proj_coords = f_position_light_space.xyz / f_position_light_space.w;

	// transform to [0,1] range
	proj_coords = proj_coords * 0.5 + 0.5;

	// get closest depth value from light's perspective (using [0,1] range f_position_light_space as coords)
	float closest_depth = texture(shadow_sampler, proj_coords.xy).r;   

	// get depth of current fragment from light's perspective
	float current_depth = proj_coords.z;

	// check whether current frag pos is in shadow
	if(current_depth > closest_depth) {
		return 1.0;
	} else {
		return 0.0;  
	}
}

vec3 calc_point_light(PointLight light, vec3 normal, vec3 f_position, vec3 view_dir) {
	vec3 light_dir = normalize(light.position - f_position);

	// diffuse shading
	float diff = max(dot(normal, light_dir), 0.0);

	// specular shading
	vec3 halfway_dir = normalize(light_dir + view_dir);
	float spec = pow(max(dot(normal, halfway_dir), 0.0), f_material_shininess);

	// attenuation
	float distance = length(light.position - f_position);
	float attenuation = 1.0 / (light.constant_ + light.linear * distance + light.quadratic * (distance * distance));    

	// combine results
	vec3 ambient = light.ambient * f_material_diffuse;
	vec3 diffuse = light.diffuse * diff * f_material_diffuse;
	vec3 specular = light.specular * spec * f_material_specular;

	ambient *= attenuation;
	diffuse *= attenuation;
	specular *= attenuation;

	float shadow = shadow_calculation(f_position_light_space);       
	return ambient + (1.0 - shadow) * (diffuse + specular);   
} 

void main() {
	// properties
	vec3 norm = normalize(f_normal);
	vec3 view_dir = normalize(camera.position - f_position);

	// phase 1: directional light
	vec3 result = calc_dir_light(light.dir_light, norm, view_dir);

	// phase 2: point lights
	for(int i = 0; i < NR_POINT_LIGHTS; i++) {
    result += calc_point_light(light.point_lights[i], norm, f_position, view_dir);   
	}

	// phase 4: texture
	out_color = vec4(result,  1.0) * texture(diffuse_sampler, f_uv);
}
