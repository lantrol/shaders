#version 430 core
// For now local size hard coded, try and add constants from python code later
layout(local_size_x=8, local_size_y=8, local_size_z=8) in;

layout(binding=0) uniform sampler3D pDivIn;
layout(binding=1) uniform sampler3D fieldIn;
layout(rgba32f, binding=2) writeonly uniform image3D fieldOut;

uniform int GRID_WIDTH;
uniform int GRID_HEIGHT;
uniform int GRID_DEPTH;

void main() {
    float h = 1.0/float(GRID_WIDTH);
    ivec3 call = ivec3(gl_GlobalInvocationID.xyz);
    if (call.x == 0 || call.x >= GRID_WIDTH-1 || call.y == 0 || call.y >= GRID_HEIGHT-1 || call.z == 0 || call.z >= GRID_DEPTH-1) {
        return;
    }
    vec4 pos_value = texelFetch(fieldIn, call, 0);
    float pL = texelFetch(pDivIn, max(min(call - ivec3(1, 0, 0), ivec3(GRID_WIDTH-2, GRID_HEIGHT-2, GRID_DEPTH-2)), ivec3(1, 1, 1)), 0).y;
    float pR = texelFetch(pDivIn, max(min(call + ivec3(1, 0, 0), ivec3(GRID_WIDTH-2, GRID_HEIGHT-2, GRID_DEPTH-2)), ivec3(1, 1, 1)), 0).y;
    float pD = texelFetch(pDivIn, max(min(call - ivec3(0, 1, 0), ivec3(GRID_WIDTH-2, GRID_HEIGHT-2, GRID_DEPTH-2)), ivec3(1, 1, 1)), 0).y;
    float pU = texelFetch(pDivIn, max(min(call + ivec3(0, 1, 0), ivec3(GRID_WIDTH-2, GRID_HEIGHT-2, GRID_DEPTH-2)), ivec3(1, 1, 1)), 0).y;
    float pF = texelFetch(pDivIn, max(min(call - ivec3(0, 0, 1), ivec3(GRID_WIDTH-2, GRID_HEIGHT-2, GRID_DEPTH-2)), ivec3(1, 1, 1)), 0).y;
    float pB = texelFetch(pDivIn, max(min(call + ivec3(0, 0, 1), ivec3(GRID_WIDTH-2, GRID_HEIGHT-2, GRID_DEPTH-2)), ivec3(1, 1, 1)), 0).y;
    float new_x = pos_value.x - 0.5*(pR - pL)/h;
    float new_y = pos_value.y - 0.5*(pU - pD)/h;
    float new_z = pos_value.z - 0.5*(pB - pF)/h;
    imageStore(fieldOut, call, vec4(new_x, new_y, new_z, texelFetch(fieldIn, call, 0).w));
}