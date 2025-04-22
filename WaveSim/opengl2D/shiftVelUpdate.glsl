#version 430 core

layout(local_size_x = 4, local_size_y = 4, local_size_z = 1) in;

layout(binding = 0) uniform sampler2D p;
layout(binding = 1) uniform sampler2D nSolidDampAtt;
layout(binding = 2) uniform sampler2D velIn;
layout(rg32f, binding = 0) writeonly uniform image2D velOut;

uniform int width;
uniform int height;
uniform float dsRhoDt;

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= width || pixel.y >= height)
        return;

    ivec2 posy = pixel + ivec2(0, 1);
    if (posy.y >= height)
        posy.y = 0;

    ivec2 posx = pixel + ivec2(1, 0);
    if (posx.x >= width)
        posx.x = 0;

    // Naming same as in python, velX is vertical and velY is horizontal
    float newVelX = texelFetch(velIn, pixel, 0).x - ((texelFetch(p, posy, 0).x - texelFetch(p, pixel, 0).x) * dsRhoDt);
    newVelX = newVelX * texelFetch(nSolidDampAtt, pixel, 0).x;

    float newVelY = texelFetch(velIn, pixel, 0).y - ((texelFetch(p, posx, 0).x - texelFetch(p, pixel, 0).x) * dsRhoDt);
    newVelY = newVelY * texelFetch(nSolidDampAtt, pixel, 0).x;

    imageStore(velOut, pixel, vec4(newVelX, newVelY, 0., 0.));
}
