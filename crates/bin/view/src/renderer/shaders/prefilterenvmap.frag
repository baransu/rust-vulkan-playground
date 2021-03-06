#version 450

layout (binding = 1) uniform samplerCube samplerEnv;

layout (binding = 2) uniform RoughnessBufferObject { 
  int numSamples;
  float roughness;
} consts;

layout(location = 0) in vec3 inPos;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

// Based omn http://byteblacksmith.com/improvements-to-the-canonical-one-liner-glsl-rand-for-opengl-es-2-0/
float random(vec2 co)
{
	float a = 12.9898;
	float b = 78.233;
	float c = 43758.5453;
	float dt= dot(co.xy ,vec2(a,b));
	float sn= mod(dt,3.14);
	return fract(sin(sn) * c);
}

vec2 hammersley2d(uint i, uint N) 
{
	// Radical inverse based on http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
	uint bits = (i << 16u) | (i >> 16u);
	bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
	bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
	bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
	bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
	float rdi = float(bits) * 2.3283064365386963e-10;
	return vec2(float(i) /float(N), rdi);
}

// Based on http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_slides.pdf
vec3 importanceSample_GGX(vec2 Xi, float roughness, vec3 normal) 
{
	// Maps a 2D point to a hemisphere with spread based on roughness
	float a = roughness * roughness;

	float phi = 2.0 * PI * Xi.x; // + random(normal.xz) * 0.1;
	float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
	float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

	vec3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;

	// Tangent space
	vec3 up = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
	vec3 tangent = normalize(cross(up, normal));
	vec3 bitangent = cross(normal, tangent);

	// Convert to world Space
	vec3 sampleVec = tangent * H.x + bitangent * H.y + normal * H.z;
	return normalize(sampleVec);
}

// Normal Distribution function
float D_GGX(float dotNH, float roughness)
{
	float alpha = roughness * roughness;
	float alpha2 = alpha * alpha;
	float denom = dotNH * dotNH * (alpha2 - 1.0) + 1.0;
	return (alpha2)/(PI * denom*denom); 
}

vec4 getEnv(vec3 uv, float mip) {
	vec4 env = textureLod(samplerEnv, uv, mip);
	return env;
}

vec3 prefilterEnvMap(vec3 R, float roughness) {
	vec3 N = R;
	vec3 V = R;
	vec3 color = vec3(0.0);
	float totalWeight = 0.0;
    const uint numSamples = uint(consts.numSamples);
	float envMapDim = float(textureSize(samplerEnv, 0).s);
	for(uint i = 0u; i < numSamples; i++) {
		vec2 Xi = hammersley2d(i, numSamples);
		vec3 H = importanceSample_GGX(Xi, roughness, N);
		vec3 L = normalize(2.0 * dot(V, H) * H - V);
		float dotNL = clamp(dot(N, L), 0.0, 1.0);

		if(dotNL > 0.0) {
			// Filtering based on https://placeholderart.wordpress.com/2015/07/28/implementation-notes-runtime-environment-map-filtering-for-image-based-lighting/

			float dotNH = clamp(dot(N, H), 0.0, 1.0);
			float dotVH = clamp(dot(V, H), 0.0, 1.0);

			// Probability Distribution Function
			float pdf = D_GGX(dotNH, roughness) * dotNH / (4.0 * dotVH) + 0.0001;
			// Slid angle of current smple
			float omegaS = 1.0 / (float(consts.numSamples) * pdf);
			// Solid angle of 1 pixel across all cube faces
			float omegaP = 4.0 * PI / (6.0 * envMapDim * envMapDim);
			// Biased (+1.0) mip level for better result
			float mipLevel = roughness == 0.0 ? 0.0 : max(0.5 * log2(omegaS / omegaP) + 1.0, 0.0f);
			color += getEnv(L, mipLevel).rgb * dotNL;
			totalWeight += dotNL;
		}
	}

	return (color / totalWeight);
}


void main()
{		
    // NOTE: we cannot normalize here!!!
    // we have to normalize in vertex shader for some reason and double normalize adds artifacts
	vec3 N = (inPos);
    vec3 color = prefilterEnvMap(N, consts.roughness);
	outColor = vec4(color.x, color.y, color.z, 1.0);
}
