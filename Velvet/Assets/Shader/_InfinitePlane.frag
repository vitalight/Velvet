#version 330

struct Material {
    float specular;
	float shininess;
};

struct SpotLight {
	vec3 position;
	vec3 direction;
	float cutOff;
	float outerCutOff;

	float constant;
	float linear;
	float quadratic;

    vec3 color;
    float ambient;
};

in VS {
    vec3 nearPoint;
    vec3 farPoint;
    vec3 normal;
} vs;

uniform mat4 _View;
uniform mat4 _Projection;
uniform mat4 _WorldToLight;

uniform vec3 _CameraPos;
uniform sampler2D _ShadowTex;
uniform SpotLight spotLight;
uniform Material material;

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

float ShadowCalculation(float ndotl, vec4 lightSpaceFragPos)
{
    // perform perspective divide
    vec3 projCoords = lightSpaceFragPos.xyz / lightSpaceFragPos.w;
    // transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
    float closestDepth = texture(_ShadowTex, projCoords.xy).r; 
    // get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    // calculate bias (based on depth map resolution and slope)
    float bias = max(0.01 * (1.0 - ndotl), 0.001);
    // check whether current frag pos is in shadow
    // float shadow = currentDepth - bias > closestDepth  ? 1.0 : 0.0;
    // PCF
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(_ShadowTex, 0);
    for(int x = -1; x <= 1; ++x)
    {
        for(int y = -1; y <= 1; ++y)
        {
            float pcfDepth = texture(_ShadowTex, projCoords.xy + vec2(x, y) * texelSize).r; 
            shadow += currentDepth - bias > pcfDepth  ? 1.0 : 0.0;        
        }    
    }
    shadow /= 9.0;
    
    // keep the shadow at 0.0 when outside the far_plane region of the light's frustum.
    if(projCoords.z > 1.0)
        shadow = 0.0;
        
    return shadow;
}

// calculates the color when using a spot light.
vec3 CalcSpotLight(SpotLight light, vec3 cameraPos, vec3 normal, vec3 worldPos, vec4 lightSpaceFragPos, Material material)
{
    vec3 lightDir = normalize(light.position - worldPos);
    // diffuse shading
    float ndotl = dot(normal, lightDir);
    float diff = max(ndotl, 0.0);
    // specular shading
	vec3 viewDir = normalize(cameraPos - worldPos);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), material.shininess);
    float distance = length(light.position - worldPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    // spotlight intensity
    float theta = dot(lightDir, normalize(-light.direction));
    float epsilon = light.cutOff - light.outerCutOff;
    float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);
    // combine results
    float ambient = light.ambient;
    float diffuse = diff;
    float specular = spec * material.specular;
    ambient *= max(0.3, attenuation * intensity);
    diffuse *= attenuation * intensity;
    specular *= attenuation * intensity;
    float shadow = ShadowCalculation(ndotl, lightSpaceFragPos);
    return light.color * (ambient + (1.0 - shadow) * (diffuse + specular));
}

vec3 ComputeDiffuseColor(vec3 worldPos)
{
    float checkerboard = checker(vec2(worldPos.x, worldPos.z), 1.0);

    return vec3(1.0 - checkerboard * 0.25);
}

vec3 GammaCorrection(vec3 color)
{
    return pow(color, vec3(1.0/2.2));
}

void main()
{
    // only draw when view ray intersects plane
	float t = -vs.nearPoint.y / (vs.farPoint.y - vs.nearPoint.y);
	if (t <= 0)
		discard;

	vec3 worldPos = vs.nearPoint + t * (vs.farPoint- vs.nearPoint);
	gl_FragDepth = computeDepth(worldPos) * 0.5 + 0.5;

    // compute color using checker-board pattern
    vec3 diffuseColor = ComputeDiffuseColor(worldPos);
    
    // lighting
	vec3 norm = vec3(0,1,0);//normalize(vs.normal);
	vec3 viewDir = normalize(_CameraPos - worldPos);
    vec4 lightSpaceFragPos = _WorldToLight * vec4(worldPos, 1.0);

	vec3 lighting = CalcSpotLight(spotLight, _CameraPos, norm, worldPos, lightSpaceFragPos, material);

	FragColor = vec4(GammaCorrection(diffuseColor * lighting), 1.0);
}