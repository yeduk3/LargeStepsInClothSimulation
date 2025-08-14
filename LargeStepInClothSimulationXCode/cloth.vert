#version 410 core

layout(location = 0) in vec3 Position;
layout(location = 2) in vec2 TexCoord;

out vec3 WPos;
out vec2 VTexCoord;

uniform mat4 M;
uniform mat4 MVP;

void main() {
    WPos = (M * vec4(Position, 1)).xyz;
    gl_Position = MVP * vec4(Position, 1);
    VTexCoord = TexCoord;
}
