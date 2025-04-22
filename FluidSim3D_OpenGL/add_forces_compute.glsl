#version 430 core
// For now local size hard coded, try and add constants from python code later
layout(local_size_x=8, local_size_y=8, local_size_z=8) in;

layout(binding=0) uniform sampler3D forcesIn;
layout(binding=1) uniform sampler3D fieldIn;
layout(rgba32f, binding=2) writeonly uniform image3D fieldOut;

uniform int GRID_WIDTH;
uniform int GRID_HEIGHT;
uniform int GRID_DEPTH;
uniform float DT;
uniform float u_time;

void main() {
    ivec3 call = ivec3(gl_GlobalInvocationID.xyz);
    if (call.x == 0 || call.x >= GRID_WIDTH-1 || call.y == 0 || call.y >= GRID_HEIGHT-1 || call.z == 0 || call.z >= GRID_DEPTH-1) {
        return;
    }
    // We need to add 0.5 to the current call ID, as the center of the Texels are at .5
    //vec3 uvs = vec3(float(call.x+0.5)/float(GRID_WIDTH), float(call.y+0.5)/float(GRID_HEIGHT), float(call.z+0.5)/float(GRID_DEPTH));
    vec4 new_dens = texelFetch(fieldIn, call, 0).xyzw + texelFetch(forcesIn, call, 0).xyzw*vec4(DT, DT, DT, DT);
    // if (u_time < 0.) {
    //     new_dens.x += texelFetch(forcesIn, call, 0).y*sin(u_time*2)*0.06;
    //     new_dens.z += texelFetch(forcesIn, call, 0).y*cos(u_time*2)*0.06;
    // }
    imageStore(fieldOut, call, new_dens);
}
