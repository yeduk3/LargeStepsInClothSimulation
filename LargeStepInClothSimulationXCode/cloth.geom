#version 410 core

layout(triangles) in;
layout(triangle_strip, max_vertices=3) out;

in vec3 WPos[];
in vec2 VTexCoord[];
out vec3 Normal;
out vec4 VPos;
out vec2 GTexCoord;

uniform mat3 normalMat;
uniform mat4 viewMat;


void main() {
    // compute normal
    vec3 normal;

    vec3 v1 = WPos[1] - WPos[0];
    vec3 v2 = WPos[2] - WPos[0];
    normal = normalize(normalMat * cross(v1, v2)); // viewCoord
    if(normal.z < 0) normal = -normal;

    Normal = normal;
    VPos = viewMat * vec4(WPos[0], 1);
    gl_Position = gl_in[0].gl_Position;
    GTexCoord = VTexCoord[0];
    EmitVertex();

    Normal = normal;
    VPos = viewMat * vec4(WPos[1], 1);
    gl_Position = gl_in[1].gl_Position;
    GTexCoord = VTexCoord[1];
    EmitVertex();

    Normal = normal;
    VPos = viewMat * vec4(WPos[2], 1);
    gl_Position = gl_in[2].gl_Position;
    GTexCoord = VTexCoord[2];
    EmitVertex();

    EndPrimitive();
}
