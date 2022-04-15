#version 330 core

struct Material {
	sampler2D diffuse;
    bool useTexture;
    vec3 tint;

    float specular;
	float smoothness;
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

// Special Begin ====
in GS {
    vec2 uv;
    vec3 centerEyePos;
} gs;

uniform mat4 _View;
uniform mat4 _Projection;
uniform mat4 _InvView;
uniform mat4 _WorldToLight;
uniform float _ParticleRadius;
// Special End ====

uniform vec3 _CameraPos;
uniform sampler2D _ShadowTex;
uniform SpotLight spotLight;
uniform Material material;

out vec4 FragColor;

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
    float bias = 0.01 * max(1.0 - ndotl, 0.1);
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
vec3 CalcSpotLight(SpotLight light, vec3 cameraPos, vec3 normal, vec3 worldPos, vec4 lightSpaceFragPos, Material material, vec3 albedo)
{
    // spotlight intensity
    vec3 lightDir = normalize(light.position - worldPos);
    float theta = dot(lightDir, normalize(-light.direction));
    float epsilon = light.cutOff - light.outerCutOff;
    float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);
    float attenuation = intensity;
    // ambient
    float ambient = light.ambient * max(0.3, attenuation);
    // diffuse
    float ndotl = dot(normal, lightDir);
    float diff = max(ndotl, 0.0);
    float diffuse = diff * (1-material.specular) * attenuation;
    // specular
	vec3 viewDir = normalize(cameraPos - worldPos);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), material.smoothness);
    float specular = spec * material.specular * attenuation;
    // shadow
    float shadow = ShadowCalculation(ndotl, lightSpaceFragPos) * 0.6;

    return light.color * (ambient * albedo + (1.0 - shadow) * (diffuse * albedo + specular));
}

vec3 GammaCorrection(vec3 color)
{
    return pow(color, vec3(1.0/2.2));
}

float ComputeDepth(vec4 pixelPos)
{
    vec4 clipSpacePos = _Projection * pixelPos;
    float depth = clipSpacePos.z / clipSpacePos.w;
    return depth * 0.5 + 0.5;
}

float ComputeLighting(vec4 pixelPos, vec3 N)
{
    vec4 lightPos = _View * vec4(spotLight.position, 1.0);
    lightPos /= lightPos.w;
    vec3 lightDir = normalize(lightPos.xyz - pixelPos.xyz);
    float diffuse = max(0.0, dot(N, lightDir))  + 0.3;
    return diffuse;
}

void main()
{
    // calculate eye-space sphere normal from texture coordinates
    vec3 N;
    N.xy = gs.uv;
    float r2 = dot(N.xy, N.xy);
    if (r2 > 1.0) discard; // kill pixels outside circle
    N.z = sqrt(1.0 -r2);

    // calculate depth
    vec4 pixelPos = vec4(gs.centerEyePos + N*_ParticleRadius, 1.0);
    gl_FragDepth = ComputeDepth(pixelPos);

    vec3 norm = (_InvView * vec4(N, 0.0)).xyz;
    vec3 worldPos = (_InvView * pixelPos).xyz;
    vec4 lightSpaceFragPos = _WorldToLight * vec4(worldPos, 1.0);
    
    vec3 diffuseColor = material.tint;
	vec3 lighting = CalcSpotLight(spotLight, _CameraPos, norm, worldPos, lightSpaceFragPos, material, diffuseColor);
	FragColor = vec4(GammaCorrection(lighting ), 1.0);
}