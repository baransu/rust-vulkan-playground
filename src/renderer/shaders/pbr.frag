#version 450

#define MAX_POINT_LIGHTS 32

// struct is shaded with gbuffer.frag
struct DirLight {
	mat4 view;
	mat4 proj;
	vec3 direction;
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
layout(binding = 5) uniform sampler2D u_depth;

layout(binding = 6) uniform samplerCube point_shadow_map;
layout(binding = 7) uniform sampler2D dir_shadow_map;


layout(binding = 8) uniform sampler2D ssao_sampler;
layout(binding = 9) uniform samplerCube samplerIrradiance;
layout(binding = 10) uniform samplerCube prefilteredMap;
layout(binding = 11) uniform sampler2D samplerBRDFLUT;  

layout(binding = 12) uniform LightUniformBufferObject { 
	PointLight point_lights[MAX_POINT_LIGHTS];
	DirLight dir_light;
	int point_lights_count;
} lights;

layout(location = 0) in vec2 f_uv;

layout(location = 0) out vec4 out_color;

#define PI 3.1415926535897932384626433832795

#define ALBEDO pow(texture(u_albedo, f_uv).xyz, vec3(2.2))

// Normal Distribution function --------------------------------------
float D_GGX(float dotNH, float roughness)
{
	float alpha = roughness * roughness;
	float alpha2 = alpha * alpha;
	float denom = dotNH * dotNH * (alpha2 - 1.0) + 1.0;
	return (alpha2)/(PI * denom*denom); 
}

// Geometric Shadowing function --------------------------------------
float G_SchlicksmithGGX(float dotNL, float dotNV, float roughness)
{
	float r = (roughness + 1.0);
	float k = (r*r) / 8.0;
	float GL = dotNL / (dotNL * (1.0 - k) + k);
	float GV = dotNV / (dotNV * (1.0 - k) + k);
	return GL * GV;
}

// Fresnel function ----------------------------------------------------
vec3 F_Schlick(float cosTheta, vec3 F0)
{
	return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}
vec3 F_SchlickR(float cosTheta, vec3 F0, float roughness)
{
	return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 prefilteredReflection(vec3 R, float roughness)
{
	const float MAX_REFLECTION_LOD = 8.0; // todo: param/const
	float lod = roughness * MAX_REFLECTION_LOD;
	// TODO: add parallax correction on R
	return textureLod(prefilteredMap, R, lod).rgb;
}

vec3 specularContribution(vec3 L, vec3 V, vec3 N, vec3 F0, float metallic, float roughness, float distance, vec3 lightColor)
{
	// Precalculate vectors and dot products	
	vec3 H = normalize (V + L);
	float dotNH = clamp(dot(N, H), 0.0, 1.0);
	float dotNV = clamp(dot(N, V), 0.0, 1.0);
	float dotNL = clamp(dot(N, L), 0.0, 1.0);

	vec3 color = vec3(0.0);

 float attenuation = 1.0 / (distance * distance);
	vec3 radiance = lightColor * attenuation;

	if (dotNL > 0.0) {
		// D = Normal distribution (Distribution of the microfacets)
		float D = D_GGX(dotNH, roughness); 
		// G = Geometric shadowing term (Microfacets shadowing)
		float G = G_SchlicksmithGGX(dotNL, dotNV, roughness);
		// F = Fresnel factor (Reflectance depending on angle of incidence)
		vec3 F = F_Schlick(dotNV, F0);		
		vec3 spec = D * F * G / (4.0 * dotNL * dotNV + 0.001);		
		vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);			
		color += (kD * ALBEDO / PI + spec) * radiance * dotNL;
	}

	return color;
}

const vec3 cubePos = vec3(0.0, 5.0, 0.0);
const vec3 cubeSize = vec3(10, 10, 10);

vec3 parallaxCorrectNormal( vec3 v) {
  vec3 nDir = normalize(v);
  vec3 rbmax = (   0.5 * ( cubeSize - cubePos ) - camera.position ) / nDir;
  vec3 rbmin = ( - 0.5 * ( cubeSize - cubePos ) - camera.position ) / nDir;

  vec3 rbminmax;
  rbminmax.x = ( nDir.x > 0. ) ? rbmax.x : rbmin.x;
  rbminmax.y = ( nDir.y > 0. ) ? rbmax.y : rbmin.y;
  rbminmax.z = ( nDir.z > 0. ) ? rbmax.z : rbmin.z;

  float correction = min(min(rbminmax.x, rbminmax.y), rbminmax.z);
  vec3 boxIntersection = camera.position + nDir * correction;

  return boxIntersection - cubePos;
}


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
float PointShadowCalculation(vec3 fragPos, vec3 lightPos) {
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
			float closestDepth = texture(point_shadow_map, (fragToLight * vec3(1, -1, 1)) + gridSamplingDisk[i] * diskRadius).r;
			closestDepth *= farPlane;   // undo mapping [0;1]
			if(currentDepth - bias > closestDepth)
					shadow += 1.0;
	}
	shadow /= float(samples);

 	return shadow;
}


float DirShadowCalculation(vec4 fragPosLightSpace, vec3 normal)
{
    // perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // transform to [0,1] range
    vec2 uv = (projCoords * 0.5 + 0.5).xy;
   
    // get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;

		float bias = 0.05; // max(0.05 * (1.0 - dot(normal, lights.dir_light.direction)), 0.005); 

		float shadow = 0.0;
		vec2 texelSize = 1.0 / textureSize(dir_shadow_map, 0);
		for(int x = -1; x <= 1; ++x)
		{
				for(int y = -1; y <= 1; ++y)
				{
						float pcfDepth = texture(dir_shadow_map, uv + vec2(x, y) * texelSize).r; 
						// NOTE: 0.8 is because we apply dir shadows to ambient
						// and I don't have global illumination so everything is super dark then
						shadow += currentDepth > pcfDepth ? 0.8 : 0.0;        
				}    
		}
		shadow /= 9.0;

		if(projCoords.z > 1.0) {
        shadow = 0.0;
		}

		return shadow;
}  

void main() {
	vec3 raw_albedo = texture(u_albedo, f_uv).xyz;
	float depth = texture(u_depth, f_uv).r;

	vec3 Normal = texture(u_normals, f_uv).xyz;
	
	vec4 RawPosition = texture(u_position, f_uv); 
	vec3 Position = RawPosition.xyz;

	vec3 color = vec3(0.0);

	// if our depth is 1.0 it means there is nothing so draw skybox
	if(depth == 1.0) {
		color = raw_albedo;
	} else {
		vec3 N = normalize(Normal);
		vec3 V = normalize(camera.position - Position);
		vec3 R = reflect(-V, N);

		vec4 metalic_roughness = texture(u_metalic_roughness, f_uv);
		float ssao = texture(ssao_sampler, f_uv).r;
		float ao = 1.0 * ssao; // metalic_roughness.r;
		float roughness = metalic_roughness.g;
		float metallic = metalic_roughness.b;

		vec3 F0 = vec3(0.04);
		F0 = mix(F0, ALBEDO, metallic);

		vec3 Lo = vec3(0.0);
		float pointLightShadows = 0;

		for(int i = 0; i < lights.point_lights_count; i++) {
			PointLight light = lights.point_lights[i];
			vec3 L = normalize(light.position - Position);
			float distance = length(light.position - Position);
			Lo += specularContribution(L, V, N, F0, metallic, roughness, distance, light.color);
			pointLightShadows += PointShadowCalculation(Position, light.position);
		}

		float NoV = max(dot(N, V), 0.0);

		vec3 F = F_SchlickR(NoV, F0, roughness);
		vec3 kD = 1.0 - F;
		kD *= 1.0 - metallic; 

		// Diffuse based on irradiance
		// TODO: add parallax correction on N
		vec3 irradiance = texture(samplerIrradiance, N).rgb;
		vec3 diffuse = irradiance * ALBEDO;	

		// Specular reflectance
		vec3 prefilteredColor = prefilteredReflection(R, roughness).rgb;	
		vec2 brdf = texture(samplerBRDFLUT, vec2(NoV, roughness)).rg;
		vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);

		vec3 ambient = (kD * diffuse + specular) * ao;

		vec4 lightSpacePosition = lights.dir_light.proj * lights.dir_light.view * RawPosition;
		float dirShadow = DirShadowCalculation(lightSpacePosition, Normal);
				
		color = (1.0 - dirShadow) * ambient + (1.0 - pointLightShadows) * Lo;
	}

	// tone mapping
	color = color / (color + vec3(1.0));
	// gamma correction
	float gamma = 2.2;
	color = pow(color, vec3(1.0/gamma));

	out_color = vec4(color, 1.0);
}
