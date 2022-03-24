#version 330

struct Material {
	sampler2D diffuse;
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

	vec3 ambient;
	vec3 diffuse;
	vec3 specular;
};

in VS {
	vec3 worldPos;
	vec3 normal;
	vec2 uv;
	vec4 lightSpacePos;
} vs;

uniform vec3 _CameraPos;
uniform sampler2D _ShadowTex;
uniform SpotLight spotLight;
uniform Material material;

out vec4 FragColor;

float ShadowCalculation(vec3 lightPos, vec4 lightSpacePos)
{
    // perform perspective divide
    vec3 projCoords = lightSpacePos.xyz / lightSpacePos.w;
    // transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
    float closestDepth = texture(_ShadowTex, projCoords.xy).r; 
    // get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    // calculate bias (based on depth map resolution and slope)
    vec3 normal = normalize(vs.normal);
    vec3 lightDir = normalize(lightPos - vs.worldPos);
    float bias = max(0.01 * (1.0 - dot(normal, lightDir)), 0.001);
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
vec3 CalcSpotLight(SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir)
{
    vec3 lightDir = normalize(light.position - fragPos);
    // diffuse shading
    float diff = max(dot(normal, lightDir), 0.0);
    // specular shading
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    // attenuation
    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    // spotlight intensity
    float theta = dot(lightDir, normalize(-light.direction));
    float epsilon = light.cutOff - light.outerCutOff;
    float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);
    // combine results
    vec3 ambient = light.ambient;
    vec3 diffuse = light.diffuse * diff;
    vec3 specular = light.specular * spec;
    ambient *= max(0.3, attenuation * intensity);
    diffuse *= attenuation * intensity;
    specular *= attenuation * intensity;
    float shadow = ShadowCalculation(light.position, vs.lightSpacePos);
    return (ambient + (1.0 - shadow) * (diffuse + specular)) * vec3(texture(material.diffuse, vs.uv));
}

void main()
{
	vec3 norm = normalize(vs.normal);
	vec3 viewDir = normalize(_CameraPos - vs.worldPos);

	vec3 result = CalcSpotLight(spotLight, norm, vs.worldPos, viewDir);

	FragColor = vec4(result, 1.0);
}

