#version 430 core
// For now local size hard coded, try and add constants from python code later
layout(local_size_x=8, local_size_y=8, local_size_z=8) in;

layout(binding=0) uniform sampler3D pDivIn;
layout(rgba32f, binding=1) writeonly uniform image3D pDivOut;

uniform int GRID_WIDTH;
uniform int GRID_HEIGHT;
uniform int GRID_DEPTH;

void main() {
    ivec3 call = ivec3(gl_GlobalInvocationID.xyz);
    if (call.x == 0 || call.x >= GRID_WIDTH-1 || call.y == 0 || call.y >= GRID_HEIGHT-1 || call.z == 0 || call.z >= GRID_DEPTH-1) {
        return;
    }
    vec4 pos_value = texelFetch(pDivIn, call, 0);
    float dC = pos_value.x;
    float pC = pos_value.y;
    float pL = texelFetch(pDivIn, max(min(call - ivec3(1, 0, 0), ivec3(GRID_WIDTH-2, GRID_HEIGHT-2, GRID_DEPTH-2)), ivec3(1, 1, 1)), 0).y;
    float pR = texelFetch(pDivIn, max(min(call + ivec3(1, 0, 0), ivec3(GRID_WIDTH-2, GRID_HEIGHT-2, GRID_DEPTH-2)), ivec3(1, 1, 1)), 0).y;
    float pD = texelFetch(pDivIn, max(min(call - ivec3(0, 1, 0), ivec3(GRID_WIDTH-2, GRID_HEIGHT-2, GRID_DEPTH-2)), ivec3(1, 1, 1)), 0).y;
    float pU = texelFetch(pDivIn, max(min(call + ivec3(0, 1, 0), ivec3(GRID_WIDTH-2, GRID_HEIGHT-2, GRID_DEPTH-2)), ivec3(1, 1, 1)), 0).y;
    float pF = texelFetch(pDivIn, max(min(call - ivec3(0, 0, 1), ivec3(GRID_WIDTH-2, GRID_HEIGHT-2, GRID_DEPTH-2)), ivec3(1, 1, 1)), 0).y;
    float pB = texelFetch(pDivIn, max(min(call + ivec3(0, 0, 1), ivec3(GRID_WIDTH-2, GRID_HEIGHT-2, GRID_DEPTH-2)), ivec3(1, 1, 1)), 0).y;
    //float pL = texelFetch(pDivIn, call - ivec3(1, 0, 0), 0).y;
    //float pR = texelFetch(pDivIn, call + ivec3(1, 0, 0), 0).y;
    //float pD = texelFetch(pDivIn, call - ivec3(0, 1, 0), 0).y;
    //float pU = texelFetch(pDivIn, call + ivec3(0, 1, 0), 0).y;
    //float pF = texelFetch(pDivIn, call - ivec3(0, 0, 1), 0).y;
    //float pB = texelFetch(pDivIn, call + ivec3(0, 0, 1), 0).y;
    float p_value = (pL + pR + pU + pD + pF + pB + dC) / 6.0;

    //p_value = p_value + texelFetch(pDivIn, call + ivec3(1, 0, 0), 0).y + texelFetch(pDivIn, call - ivec3(1, 0, 0), 0).y;
    //p_value = p_value + texelFetch(pDivIn, call + ivec3(0, 1, 0), 0).y + texelFetch(pDivIn, call - ivec3(0, 1, 0), 0).y;
    //p_value = p_value + texelFetch(pDivIn, call + ivec3(0, 0, 1), 0).y + texelFetch(pDivIn, call - ivec3(0, 0, 1), 0).y;
    //p_value = p_value/6.0;
    imageStore(pDivOut, call, vec4(pos_value.x, p_value, pos_value.z, pos_value.w));
}