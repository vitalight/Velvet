#version 330

in VS {
    vec3 nearPoint;
    vec3 farPoint;
} vs;

uniform mat4 _View;
uniform mat4 _Projection;

out vec4 FragColor;

float computeDepth(vec3 pos) {
    vec4 clip_space_pos = _Projection * _View * vec4(pos.xyz, 1.0);
    return (clip_space_pos.z / clip_space_pos.w);
}

float computeLinearDepth(vec3 pos) {
    float near = 0.01;
    float far = 100;

    vec4 clip_space_pos = _Projection * _View * vec4(pos.xyz, 1.0);
    float clip_space_depth = (clip_space_pos.z / clip_space_pos.w) * 2.0 - 1.0; // put back between -1 and 1
    float linearDepth = (2.0 * near * far) / (far + near - clip_space_depth * (far - near)); // get linear value between 0.01 and 100
    return linearDepth / far; // normalize
}

float checker(vec2 uv, float repeats)
{
    float cx = floor(repeats * uv.x);
    float cy = floor(repeats * uv.y); 
    float result = mod(cx + cy, 2.0);
    return sign(result);
}

void main()
{
//	FragColor = vec4(1.0, 0.0, 0.0, 1.0); // set all 4 vector values to 1.0
	float t = -vs.nearPoint.y / (vs.farPoint.y - vs.nearPoint.y);
	if (t <= 0)
		discard;
	vec3 fragPos3D = vs.nearPoint + t * (vs.farPoint- vs.nearPoint);
	gl_FragDepth = computeDepth(fragPos3D) * 0.5 + 0.5;

    float linearDepth = computeLinearDepth(fragPos3D);
    float fading = max(0, (0.5 - linearDepth));
    vec3 color = vec3(fract(fragPos3D.x), 0.0, fract(fragPos3D.z));
    float checkerboard = checker(vec2(fragPos3D.x, fragPos3D.z), 1.0);
    FragColor = vec4(vec3(0.95 - checkerboard * 0.2), 1.0); // opacity = 1 when t > 0, opacity = 0 otherwise
}