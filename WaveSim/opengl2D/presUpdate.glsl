#version 430 core

layout(local_size_x = 4, local_size_y = 4, local_size_z = 1) in;

layout(binding = 0) uniform sampler2D p;
layout(binding = 1) uniform sampler2D attDt;
layout(binding = 2) uniform sampler2D velIn;
layout(r32f, binding = 0) writeonly uniform image2D pOut;

uniform int width;
uniform int height;
uniform float dsCCRhoDt;

void main() {
    // vShiftNx = np.roll(self.vx, 1, axis=0)
    // vShiftNy = np.roll(self.vy, 1, axis=1)
    // self.p = np.float32((self.p - ((self.vx - vShiftNx) + (self.vy - vShiftNy)) * dsCCRhoDt) / self.attDt)

    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= width || pixel.y >= height)
        return;

    ivec2 posy = pixel - ivec2(0, 1);
    if (posy.y == 0)
        posy.y = height - 1;

    ivec2 posx = pixel - ivec2(1, 0);
    if (posx.x == 0)
        posx.x = width - 1;

    // Naming same as in python, velX is vertical and velY is horizontal
    float dvx = texelFetch(velIn, pixel, 0).x - texelFetch(velIn, posy, 0).x;
    float dvy = texelFetch(velIn, pixel, 0).y - texelFetch(velIn, posx, 0).y;

    float newP = (dvx + dvy) * dsCCRhoDt;
    newP = texelFetch(p, pixel, 0).x - newP;
    newP = newP / texelFetch(attDt, pixel, 0).x;
    imageStore(pOut, pixel, vec4(newP, 0., 0., 0.));
}
