#version 430 core

layout(local_size_x = 4, local_size_y = 4, local_size_z = 1) in;

layout(binding = 0) uniform sampler2D p;
layout(binding = 1) uniform sampler2D ampPhaseIn;
layout(rg32f, binding = 0) writeonly uniform image2D ampPhaseOut;

#define TWOPI 6.2831853072

uniform int width;
uniform int height;
uniform float timeStamp;
uniform float T;

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= width || pixel.y >= height)
        return;

    if (texelFetch(p, pixel, 0).x > texelFetch(ampPhaseIn, pixel, 0).x) {
        float newAmp = texelFetch(p, pixel, 0).x;
        float newPhase = mod(timeStamp, T) * (TWOPI / T);
        imageStore(ampPhaseOut, pixel, vec4(newAmp, newPhase, 0., 0.));
    }
    else {
        imageStore(ampPhaseOut, pixel, vec4(texelFetch(ampPhaseIn, pixel, 0).x, texelFetch(ampPhaseIn, pixel, 0).y, 0., 0.));
    }
}
