#version 430 core
// For now local size hard coded, try and add constants from python code later
layout(local_size_x=8, local_size_y=8, local_size_z=8) in;

layout(binding=0) uniform sampler3D fieldIn;
layout(rgba32f, binding=1) writeonly uniform image3D pDivOut;

uniform int GRID_WIDTH;
uniform int GRID_HEIGHT;
uniform int GRID_DEPTH;

void main() {
    float h = 1.0/float(GRID_WIDTH);
    ivec3 call = ivec3(gl_GlobalInvocationID.xyz);
    if (call.x == 0 || call.x >= GRID_WIDTH-1 || call.y == 0 || call.y >= GRID_HEIGHT-1 || call.z == 0 || call.z >= GRID_DEPTH-1) {
        return;
    }
    //float vL = texelFetch(fieldIn, max(min(call - ivec3(1, 0, 0), ivec3(GRID_WIDTH-2, GRID_HEIGHT-2, GRID_DEPTH-2)), ivec3(1, 1, 1)), 0).x;
    //float vR = texelFetch(fieldIn, max(min(call + ivec3(1, 0, 0), ivec3(GRID_WIDTH-2, GRID_HEIGHT-2, GRID_DEPTH-2)), ivec3(1, 1, 1)), 0).x;
    //float vD = texelFetch(fieldIn, max(min(call - ivec3(0, 1, 0), ivec3(GRID_WIDTH-2, GRID_HEIGHT-2, GRID_DEPTH-2)), ivec3(1, 1, 1)), 0).y;
    //float vU = texelFetch(fieldIn, max(min(call + ivec3(0, 1, 0), ivec3(GRID_WIDTH-2, GRID_HEIGHT-2, GRID_DEPTH-2)), ivec3(1, 1, 1)), 0).y;
    //float vF = texelFetch(fieldIn, max(min(call - ivec3(0, 0, 1), ivec3(GRID_WIDTH-2, GRID_HEIGHT-2, GRID_DEPTH-2)), ivec3(1, 1, 1)), 0).z;
    //float vB = texelFetch(fieldIn, max(min(call + ivec3(0, 0, 1), ivec3(GRID_WIDTH-2, GRID_HEIGHT-2, GRID_DEPTH-2)), ivec3(1, 1, 1)), 0).z;
    
    float vL = texelFetch(fieldIn, call - ivec3(1, 0, 0), 0).x;
    float vR = texelFetch(fieldIn, call + ivec3(1, 0, 0), 0).x;
    float vD = texelFetch(fieldIn, call - ivec3(0, 1, 0), 0).y;
    float vU = texelFetch(fieldIn, call + ivec3(0, 1, 0), 0).y;
    float vF = texelFetch(fieldIn, call - ivec3(0, 0, 1), 0).z;
    float vB = texelFetch(fieldIn, call + ivec3(0, 0, 1), 0).z;

    // float div_value = texelFetch(fieldIn, call + ivec3(1, 0, 0), 0).x - texelFetch(fieldIn, call - ivec3(1, 0, 0), 0).x;
    // div_value = div_value + texelFetch(fieldIn, call + ivec3(0, 1, 0), 0).y - texelFetch(fieldIn, call - ivec3(0, 1, 0), 0).y;
    // div_value = div_value + texelFetch(fieldIn, call + ivec3(0, 0, 1), 0).z - texelFetch(fieldIn, call - ivec3(0, 0, 1), 0).z;
    // div_value = -0.5*h*div_value;
    
    float div_value = (vR-vL+vU-vD+vB-vF)*-0.5*h;
    imageStore(pDivOut, call, vec4(div_value, 0., 0., 0.));
}