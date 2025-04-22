#version 430 core

layout(local_size_x = 4, local_size_y = 4, local_size_z = 1) in;

layout(binding = 0) uniform sampler2D pIn;
layout(rg32f, binding = 0) writeonly uniform image2D pOut;

uniform int width;
uniform int height;
uniform int positions;

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= width || pixel.y >= height)
        return;

    ivec2 posx = pixel + ivec2(-positions, 0);
    if (posx.x >= width)
        posx.x = 0;

    imageStore(pOut, pixel, vec4(texelFetch(pIn, posx, 0).x, 0., 0., 0.));
}
