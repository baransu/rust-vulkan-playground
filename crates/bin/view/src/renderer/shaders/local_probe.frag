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
	vec3 color;
	// NOTE: is constant reserved keyword?
	float constant_;
	float linear;
	float quadratic;	
};

// duplicated definition in model.vert
layout(set = 0, binding = 1) uniform CameraUniformBufferObject {
	mat4 view;
	mat4 proj;
	vec3 position;
} camera;

// The `color_input` parameter of the `draw` method.
layout(set = 1, binding = 0) uniform sampler2D diffuse_sampler;
layout(set = 1, binding = 1) uniform sampler2D normal_sampler;
layout(set = 1, binding = 2) uniform sampler2D metalic_roughness_sampler;

layout(set = 2, binding = 0) uniform LightUniformBufferObject { 
	PointLight point_lights[MAX_POINT_LIGHTS];
	int point_lights_count;
} lights;

layout(location = 0) in vec2 f_uv;
layout(location = 1) in vec3 f_normal;
layout(location = 2) in vec4 f_tangent;
layout(location = 3) in vec3 f_position;

layout(location = 0) out vec4 out_color;

const float PI = 3.14159265359;

vec3 fresnelShlick(float cosTheta,  vec3 F0);
float DistributionGGX(vec3 N, vec3 H, float roughness);
float GeometryShlickGGX(float NdotV, float roughness); 
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness);

vec3 getNormal() {
	vec3 tangentNormal = texture(normal_sampler, f_uv).xyz * 2.0 - 1.0;

	vec3 N  = normalize(f_normal);
	vec3 T  = normalize(f_tangent.xyz);
	vec3 B  = normalize(cross(N, T));
	mat3 TBN = mat3(T, B, N);

	return normalize(TBN * tangentNormal);	
}

void main() {
	vec3 albedo = pow(texture(diffuse_sampler, f_uv).rgb, vec3(2.2));

	vec3 Normal = getNormal();

	vec3 Position = f_position;

	vec3 N = normalize(Normal);
	vec3 V = normalize(camera.position - Position);

	vec4 metalic_roughness = texture(metalic_roughness_sampler, f_uv);
	float roughness = metalic_roughness.g;
	float metallic = 0.0; // metalic_roughness.b;

	vec3 F0 = vec3(0.04);
	F0 = mix(F0, albedo, metallic);

	vec3 Lo = vec3(0.0);
	for(int i = 0; i < lights.point_lights_count; i++) {
		PointLight light = lights.point_lights[i];
		vec3 L = normalize(light.position - Position);
		vec3 H = normalize(V + L);

		float distance = length(light.position - Position);
		float attenuation = 1.0 / (light.constant_ + light.linear * distance + light.quadratic * distance * distance);
		vec3 radiance = light.color * attenuation;

		vec3 F = fresnelShlick(max(dot(H, V), 0.0), F0);

		float NDF = DistributionGGX(N, H, roughness);
		float G = GeometrySmith(N, V, L, roughness);

		vec3 numerator = NDF * G * F;
		float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
		vec3 specular = numerator / denominator;

		vec3 kS = F;
		vec3 kD = vec3(1.0) - kS;

		kD *= 1.0 - metallic;

		float NdotL = max(dot(N, L), 0.0);
		Lo += (kD * albedo / PI + specular) * radiance * NdotL;
	}

	vec3 ambient = vec3(0.03) * albedo;
	vec3 color = ambient + Lo;

	// gamma correction and tone mapping
	float gamma = 2.2;
	color = color / (color + vec3(1.0));
	color = pow(color, vec3(1.0/gamma));

	out_color = vec4(color, 1.0);
}

vec3 fresnelShlick(float cosTheta,  vec3 F0) {
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
