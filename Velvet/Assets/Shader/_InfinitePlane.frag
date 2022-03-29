#version 330

struct Material {
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

in VS {
    vec3 nearPoint;
    vec3 farPoint;
} vs;

uniform mat4 _View;
uniform mat4 _Projection;
uniform mat4 _WorldToLight;

uniform vec3 _CameraPos;
uniform vec4 _Plane;
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
    float bias = 0.0;//max(0.01 * (1.0 - ndotl), 0.001);
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

float ComputeFragDepth(vec3 pos) {
    vec4 clip_space_pos = _Projection * _View * vec4(pos.xyz, 1.0);
    return (clip_space_pos.z / clip_space_pos.w)  * 0.5 + 0.5;
}

float filterwidth(vec2 v)
{
  vec2 fw = max(abs(dFdx(v)), abs(dFdy(v)));
  return max(fw.x, fw.y);
}

vec2 bump(vec2 x) 
{
	return (floor((x)/2) + 2.f * max(((x)/2) - floor((x)/2) - .5f, 0.f)); 
}

float checker(vec2 uv)
{
  float width = filterwidth(uv);
  vec2 p0 = uv - 0.5 * width;
  vec2 p1 = uv + 0.5 * width;
  
  vec2 i = (bump(p1) - bump(p0)) / width;
  return i.x * i.y + (1 - i.x) * (1 - i.y);
}

vec3 ComputeDiffuseColor(vec3 worldPos)
{
    float checkerboard = checker(vec2(worldPos.x, worldPos.z) * 2.0);
    return vec3(1.0- checkerboard * 0.25);
}

vec3 ApplyFog(vec3 color, vec3 pos)
{
    vec4 viewSpacePos = _View * vec4(pos, 1.0);
    return mix(vec3(0.0), color, exp(viewSpacePos.z * 0.05));
}

vec3 GammaCorrection(vec3 color)
{
    return pow(color, vec3(1.0/2.2));
}

void main()
{
    // only draw when view ray intersects plane
    vec3 planeNormal = _Plane.xyz;
    float planeDistance = _Plane.w;
    vec3 rayDirection = vs.farPoint - vs.nearPoint;
    vec3 rayOrigin = vs.nearPoint;

    float t = -(planeDistance + dot(rayOrigin, planeNormal)) / dot(rayDirection, planeNormal);
	if (t <= 0)
		discard;

	vec3 worldPos = rayOrigin + t * rayDirection;
	gl_FragDepth = ComputeFragDepth(worldPos);

    // lighting
    vec3 diffuseColor = ComputeDiffuseColor(worldPos);
	vec3 viewDir = normalize(_CameraPos - worldPos);
    vec4 lightSpaceFragPos = _WorldToLight * vec4(worldPos, 1.0);
	vec3 lighting = CalcSpotLight(spotLight, _CameraPos, planeNormal, worldPos, lightSpaceFragPos, material, diffuseColor);
    vec3 finalColor = GammaCorrection(ApplyFog(lighting, worldPos));

	FragColor = vec4(finalColor, 1.0);
}