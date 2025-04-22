#version 430 core

layout(local_size_x = 4, local_size_y = 4, local_size_z = 1) in;

layout(binding = 0) uniform sampler2D att;
layout(r32f, binding = 0) writeonly uniform image2D attDt;

uniform int width;
uniform int height;
uniform float dt;

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= width || pixel.y >= height)
        return;
    float result = (1. + texelFetch(att, pixel, 0).x * dt);
    imageStore(attDt, pixel, vec4(result, 0., 0., 0.));
}
