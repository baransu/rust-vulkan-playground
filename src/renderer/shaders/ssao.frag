#version 450

#define SAMPLES_SIZE 64
#define SSAO_RADIUS 0.5

// duplicated definition in model.vert
layout(binding = 0) uniform CameraUniformBufferObject {
	mat4 view;
	mat4 proj;
	vec3 position;
} camera;

layout(binding = 1) uniform sampler2D u_position;
layout(binding = 2) uniform sampler2D u_normals;
layout(binding = 3) uniform sampler2D noise_sampler;

layout(binding = 4) uniform SsaoUniformBufferObject {
	vec4 samples[SAMPLES_SIZE];
} ssao;

// in
layout(location = 0) in vec2 f_uv;

// out
layout(location = 0) out float out_color;

void main() {
	vec3 f_position = texture(u_position, f_uv).xyz;
	vec3 f_normal = normalize(texture(u_normals, f_uv).rgb * 2.0 - 1.0);

	ivec2 tex_dim = textureSize(u_position, 0);
	ivec2 noise_dim = textureSize(noise_sampler, 0);
	const vec2 noise_uv = vec2(float(tex_dim.x)/float(noise_dim.x), float(tex_dim.y)/(noise_dim.y)) * f_uv;  

	vec3 random_vec = texture(noise_sampler, noise_uv).xyz * 2.0 - 1.0;

	vec3 tangent   = normalize(random_vec - f_normal * dot(random_vec, f_normal));
	vec3 bitangent = cross(f_normal, tangent);
	mat3 TBN       = mat3(tangent, bitangent, f_normal);  

	float occlusion = 0.0;
	
	const float bias = 0.025f;

	for(int i = 0; i < SAMPLES_SIZE; i++) {
			// get sample position
			vec3 sample_pos = TBN * ssao.samples[i].xyz; // from tangent to view-space
			sample_pos = f_position + sample_pos * SSAO_RADIUS; 

			vec4 offset = vec4(sample_pos, 1.0);
			offset      = camera.proj * offset;   // from view to clip-space
			offset.xyz /= offset.w;               // perspective divide
			offset.xyz  = offset.xyz * 0.5 + 0.5; // transform to range 0.0 - 1.0  

			float sample_depth = -texture(u_position, offset.xy).w; 
			
			float range_check = smoothstep(0.0, 1.0, SSAO_RADIUS / abs(f_position.z - sample_depth));
			occlusion += (sample_depth >= sample_pos.z + bias ? 1.0 : 0.0) * range_check;  
	}  

	occlusion = 1.0 - (occlusion / SAMPLES_SIZE);

	out_color = occlusion;
}
