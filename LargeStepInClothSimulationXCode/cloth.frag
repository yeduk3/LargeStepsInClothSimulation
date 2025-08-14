#version 410 core

in vec3 Normal;
in vec4 VPos;
in vec2 GTexCoord;

out vec4 FragColor;

uniform vec3 lightPosition;
uniform mat4 viewMat;
uniform vec3 lightColor;
uniform vec3 diffuseColor;
uniform vec3 specularColor;
uniform float shininess;

uniform sampler2D fabricTex;


vec4 phong() {
    
    vec3 materialColor = texture(fabricTex, GTexCoord).rgb;
//    vec3 materialColor = diffuseColor;
    
    vec4 lp = viewMat * vec4(lightPosition, 1);
    vec3 l = lp.xyz/lp.w - VPos.xyz/VPos.w;
    vec3 L = normalize(l);
    vec3 N = normalize(Normal);
    vec3 R = 2 * dot(L, N) * N - L;
    vec3 I = lightColor / dot(l, l);

    vec3 ambient = materialColor * vec3(0.1);
    vec3 diffuse = I * materialColor * max(dot(L, N), 0);
    vec3 specular = I * specularColor * pow(max(R.z, 0), shininess);

    vec3 color = ambient + diffuse + specular;

    return vec4(pow(color, vec3(1/2.2)), 1);
}

void main() {
    FragColor = phong();
}
