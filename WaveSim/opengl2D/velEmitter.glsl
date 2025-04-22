#version 430 core

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly buffer xxInData {
    int XX[];
} xxData;

layout(std430, binding = 1) readonly buffer yyInData {
    int YY[];
} yyData;

layout(rg32f, binding = 0) writeonly uniform image2D velOut;

uniform float velX;
uniform float velY;

void main() {
    int invocIndex = int(gl_GlobalInvocationID.x);
    ivec2 updatePos = ivec2(yyData.YY[invocIndex], xxData.XX[invocIndex]);
    imageStore(velOut, updatePos, vec4(velX, velY, 0., 0.));
}
