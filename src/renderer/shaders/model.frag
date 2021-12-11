#version 450

// duplicated definition in model.vert too
layout(binding = 0) uniform CameraUniformBufferObject {
	mat4 view;
	mat4 proj;
	vec3 position;
} camera;

layout(binding = 2) uniform sampler2D tex_sampler;

layout(binding = 3) uniform DirectionLightUniformBufferObject { 
	vec3 direction;
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;
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

// out
layout(location = 0) out vec4 out_color;

void main() {
	vec3 norm = normalize(f_normal);
	// vec3 light_dir = normalize(light.position - f_position);
	vec3 light_dir = normalize(-light.direction); 

	// ambient light
	vec3 ambient = f_material_ambient * light.ambient;

	// diffuse light
	float diff = max(dot(norm, light_dir), 0.0);
	vec3 diffuse = diff * f_material_diffuse * light.diffuse;

	// specular light
	vec3 view_dir = normalize(camera.position - f_position);
	vec3 reflect_dir = reflect(-light_dir, norm);  

	float spec = pow(max(dot(view_dir, reflect_dir), 0.0), f_material_shininess);
	vec3 specular = spec * f_material_specular * light.specular;  

	// diff + specular + ambient
	vec3 result = (ambient + diffuse + specular) * texture(tex_sampler, f_uv).xyz;
	out_color = vec4(result,  1.0);
}
