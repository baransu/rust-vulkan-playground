#version 450

#define MAX_POINT_LIGHTS 32

struct DirectionalLight { 
	vec3 direction;
	vec3 color;
};

struct PointLight {
	vec3 position;
	vec3 color;
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
layout(binding = 7) uniform samplerCube irradianceMap;
layout(binding = 8) uniform samplerCube prefilterMap;
layout(binding = 9) uniform sampler2D   brdfLUT;  

// duplicated definition in model.vert and shaders.rs
layout(binding = 10)	uniform LightSpaceUniformBufferObject {
	mat4 matrix;
} light_space;

layout(binding = 11) uniform LightUniformBufferObject { 
	PointLight point_lights[MAX_POINT_LIGHTS];
	DirectionalLight dir_light;
	int point_lights_count;
} lights;

layout(location = 0) in vec2 f_uv;

layout(location = 0) out vec4 out_color;

#define PI 3.1415926535897932384626433832795

vec3 fresnelSchlick(float cosTheta,  vec3 F0);
float DistributionGGX(vec3 N, vec3 H, float roughness);
float GeometryShlickGGX(float NdotV, float roughness); 
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness);

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}  

void main() {
	vec3 raw_albedo = texture(u_albedo, f_uv).xyz;
	vec3 albedo = pow(raw_albedo, vec3(2.2));

	vec3 Normal = texture(u_normals, f_uv).xyz;
	vec3 Position = texture(u_position, f_uv).xyz;

	vec3 color = vec3(0.0);

	// if our depth is 0.0 it means there is nothing so return skybox
	if(texture(u_position, f_uv).w == 0.0) {
		color = raw_albedo;
	} else {
		vec3 N = normalize(Normal);
		vec3 V = normalize(camera.position - Position);
		vec3 R = reflect(-V, N);

		vec4 metalic_roughness = texture(u_metalic_roughness, f_uv);
		// TODO: fix ssao so we don't need to load ao from texture
		float ao = 1.0; // metalic_roughness.r;
		float metallic = metalic_roughness.g;
		float roughness = metalic_roughness.b;

		vec3 F0 = vec3(0.04);
		F0 = mix(F0, albedo, metallic);

		vec3 Lo = vec3(0.0);
		for(int i = 0; i < lights.point_lights_count; i++) {
			PointLight light = lights.point_lights[i];
			vec3 L = normalize(light.position - Position);
			vec3 H = normalize(V + L);

			float distance = length(light.position - Position);
			float attenuation = 1.0 / (distance * distance);
			vec3 radiance = light.color * attenuation;

			float NDF = DistributionGGX(N, H, roughness);
			float G = GeometrySmith(N, V, L, roughness);
			vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

			vec3 numerator = NDF * G * F;
			float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
			vec3 specular = numerator / denominator;

			vec3 kS = F;
			vec3 kD = vec3(1.0) - kS;

			kD *= 1.0 - metallic;

			float NdotL = max(dot(N, L), 0.0);

			Lo += (kD * albedo / PI + specular) * radiance * NdotL;
		}

		vec3 F = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
		vec3 kS = F;
		vec3 kD = 1.0 - kS;
		kD *= 1.0 - metallic;

    vec3 irradiance = texture(irradianceMap, N).rgb;
    vec3 diffuse    = irradiance * albedo;

		 const float MAX_REFLECTION_LOD = 4.0;
    vec3 prefilteredColor = textureLod(prefilterMap, R,roughness * MAX_REFLECTION_LOD).rgb;    
		vec2 brdf = texture(brdfLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
		vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);

		vec3 ambient = (kD * diffuse + specular) * ao;

		color = ambient + Lo;
	}

	// tone mapping
	color = color / (color + vec3(1.0));
	// gamma correction
	float gamma = 2.2;
	color = pow(color, vec3(1.0/gamma));

	out_color = vec4(color, 1.0);
}

vec3 fresnelSchlick(float cosTheta,  vec3 F0) {
	return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float DistributionGGX(vec3 N, vec3 H, float roughness) {
	float a = roughness * roughness;
	float a2 = a * a;
	float NdotH = max(dot(N, H), 0.0);
	float NdotH2 = NdotH * NdotH;

	float nom = a2;
	float denom = (NdotH2 * (a2 - 1.0) + 1.0);
	denom = PI * denom * denom;

	return nom / denom;
}

float GeometryShlickGGX(float NdotV, float roughness) {
	float r = roughness + 1.0;
	float k = (r * r) / 8.0;

	float nom = NdotV;
	float denom = NdotV * (1.0 - k) + k;

	return nom / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
	float NdotV = max(dot(N, V), 0.0);
	float NdotL = max(dot(N, L), 0.0);
	float ggx2 = GeometryShlickGGX(NdotV, roughness);
	float ggx1 = GeometryShlickGGX(NdotL, roughness);

	return ggx1 * ggx2;
}
