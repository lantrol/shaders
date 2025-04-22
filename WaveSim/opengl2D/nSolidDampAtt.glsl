#version 430 core

layout(local_size_x = 4, local_size_y = 4, local_size_z = 1) in;

layout(binding = 0) uniform sampler2D solid;
layout(binding = 1) uniform sampler2D att;
layout(r32f, binding = 2) writeonly uniform image2D nSolidDampAtt;

uniform int width;
uniform int height;
uniform float damp;
uniform float dt;

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= width || pixel.y >= height)
        return;
    float result = (1. - texelFetch(solid, pixel, 0).x) * damp / (1 + texelFetch(att, pixel, 0).x * dt);
    imageStore(nSolidDampAtt, pixel, vec4(result, 0., 0., 0.));
}
