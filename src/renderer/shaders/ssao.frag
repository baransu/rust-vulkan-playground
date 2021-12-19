#version 450

#define SAMPLES_SIZE 64

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
layout(location = 0) out vec4 out_color;

const float radius = 0.5;
const float bias = 0.025;

void main() {
	vec3 position = texture(u_position, f_uv).rgb;

	// NOTE: G-Buffer normals are world space so we have to move them to view space
	vec3 normal = normalize(camera.view * (texture(u_normals, f_uv) * 2.0 - 1.0)).rgb;

	ivec2 tex_dim = textureSize(u_position, 0);
	ivec2 noise_dim = textureSize(noise_sampler, 0);
	const vec2 noise_uv = vec2(float(tex_dim.x)/float(noise_dim.x), float(tex_dim.y)/(noise_dim.y));  

	vec3 random = normalize(texture(noise_sampler, f_uv * noise_uv) * 2.0 - 1.0).xyz;

	vec3 tangent   = normalize(random - normal * dot(random, normal));
	vec3 bitangent = cross(normal, tangent);
	mat3 TBN       = mat3(tangent, bitangent, normal);  

	float occlusion = 0.0;
	for(int i = 0; i < SAMPLES_SIZE; i++) {
			// get sample position
			vec3 sample_pos = TBN * ssao.samples[i].xyz; // from tangent to view-space
			sample_pos = position + sample_pos * radius; 

			vec4 offset = vec4(sample_pos, 1.0);
			offset      = camera.proj * offset;   // from view to clip-space
			offset.xyz /= offset.w;               // perspective divide
			offset.xyz  = offset.xyz * 0.5 + 0.5; // transform to range 0.0 - 1.0  

			float sample_depth = texture(u_position, offset.xy).z; 
			
			float range_check = smoothstep(0.0, 1.0, radius / abs(position.z - sample_depth));
			occlusion += (sample_depth >= sample_pos.z + bias ? 1.0 : 0.0) * range_check;  
	}  

	float final = 1.0 - (occlusion / SAMPLES_SIZE);

	out_color = vec4(final, final, final, 1.0);
}
