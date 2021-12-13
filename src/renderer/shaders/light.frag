#version 450

struct DirectionalLight { 
	vec3 direction;
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;
};

// duplicated definition in model.vert
layout(binding = 0) uniform CameraUniformBufferObject {
	mat4 view;
	mat4 proj;
	vec3 position;
} camera;

// The `color_input` parameter of the `draw` method.
layout(input_attachment_index = 0, set = 0, binding = 1) uniform subpassInput u_position;
layout(input_attachment_index = 0, set = 0, binding = 2) uniform subpassInput u_normals;
layout(input_attachment_index = 0, set = 0, binding = 3) uniform subpassInput u_albedo;

layout(location = 4) in vec3 f_material_ambient;
layout(location = 5) in vec3 f_material_diffuse;
layout(location = 6) in vec3 f_material_specular;
layout(location = 7) in float f_material_shininess;

layout(location = 0) out vec4 f_color;

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

void main() {
	vec3 ambient = vec3(1.0, 1.0, 1.0);

	DirectionalLight light = DirectionalLight(vec3(-0.2, -1.0, -0.3), vec3(0.25), vec3(0.25), vec3(0.0));

	vec3 f_position = subpassLoad(u_position).xyz;
	vec3 normal = subpassLoad(u_normals).rgb;
	vec3 view_dir = normalize(camera.position - f_position);

	vec3 lighting = calc_dir_light(light, normal, view_dir) + (ambient * 0.0); // har

	f_color = vec4(lighting * subpassLoad(u_albedo).rgb, 1.0);
}
