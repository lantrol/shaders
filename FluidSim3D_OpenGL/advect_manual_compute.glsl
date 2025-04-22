#version 430 core
// For now local size hard coded, try and add constants from python code later
layout(local_size_x=8, local_size_y=8, local_size_z=8) in;

layout(binding=0) uniform sampler3D fieldIn;
layout(rgba32f, binding=1) writeonly uniform image3D fieldOut;

uniform int GRID_WIDTH;
uniform int GRID_HEIGHT;
uniform int GRID_DEPTH;
uniform float DT;

void main() {
    ivec3 call = ivec3(gl_GlobalInvocationID.xyz);
    // if (call.x == 0 || call.x >= GRID_WIDTH-1 || call.y == 0 || call.y >= GRID_HEIGHT-1 || call.z == 0 || call.z >= GRID_DEPTH-1) {
    //     imageStore(fieldOut, call, texelFetch(fieldIn, call, 0).xyzw);
    //     return;
    // }
    
    if (call.x >= GRID_WIDTH || call.y >= GRID_HEIGHT || call.z >= GRID_DEPTH) {
        return;
    }

    if (call.x < 1) {
        vec3 value = texelFetch(fieldIn, ivec3(1, call.y, call.z), 0).xyz;
        imageStore(fieldOut, call, vec4(-value.x, value.y, value.z, 0.));
        return;
    }
    else if (call.x >= GRID_WIDTH-1) {
        vec3 value = texelFetch(fieldIn, ivec3(GRID_WIDTH-2, call.y, call.z), 0).xyz;
        imageStore(fieldOut, call, vec4(-value.x, value.y, value.z, 0));
        return;
    }
    else if (call.y < 1) {
        vec3 value = texelFetch(fieldIn, ivec3(call.x, 1, call.z), 0).xyz;
        imageStore(fieldOut, call, vec4(value.x, -value.y, value.z, 0));
        return;
    }
    else if (call.y >= GRID_HEIGHT-1) {
        vec3 value = texelFetch(fieldIn, ivec3(call.x, GRID_HEIGHT-2, call.z), 0).xyz;
        imageStore(fieldOut, call, vec4(value.x, -value.y, value.z, 0));
        return;
    }
    else if (call.z < 1) {
        vec3 value = texelFetch(fieldIn, ivec3(call.x, call.y, 1), 0).xyz;
        imageStore(fieldOut, call, vec4(value.x, value.y, -value.z, 0));
        return;
    }
    else if (call.z >= GRID_DEPTH-1) {
        vec3 value = texelFetch(fieldIn, ivec3(call.x, call.y, GRID_DEPTH-2), 0).xyz;
        imageStore(fieldOut, call, vec4(value.x, value.y, -value.z, 0));
        return;
    }

    float dt0 = float(DT)*GRID_WIDTH;
    vec3 vel = texelFetch(fieldIn, call, 0).xyz;
    
    float x, y, z;
    int i0, i1, j0, j1, k0, k1;
    float s0, s1, t0, t1, u0, u1;

    x = min(max(call.x - dt0*vel.x, 0.5), GRID_WIDTH-1.5);
    y = min(max(call.y - dt0*vel.y, 0.5), GRID_HEIGHT-1.5);
    z = min(max(call.z - dt0*vel.z, 0.5), GRID_DEPTH-1.5);

    i0 = int(x);
    i1 = i0 + 1;
    j0 = int(y);
    j1 = j0 + 1;
    k0 = int(z);
    k1 = k0 + 1;

    s1 = x - float(i0);
    s0 = 1.0 - s1;
    t1 = y - float(j0);
    t0 = 1.0 - t1;
    u1 = z - float(k0);
    u0 = 1.0 - u1;

    vec4 new_vel;
    vec4 temp = t0*(u0*texelFetch(fieldIn, ivec3(i0, j0, k0), 0) + u1*texelFetch(fieldIn, ivec3(i0, j0, k1), 0));
    temp += t1*(u0*texelFetch(fieldIn, ivec3(i0, j1, k0), 0) + u1*texelFetch(fieldIn, ivec3(i0, j1, k1), 0));
    new_vel = s0*temp;

    temp = t0*(u0*texelFetch(fieldIn, ivec3(i1, j0, k0), 0) + u1*texelFetch(fieldIn, ivec3(i1, j0, k1), 0));
    temp += t1*(u0*texelFetch(fieldIn, ivec3(i1, j1, k0), 0) + u1*texelFetch(fieldIn, ivec3(i1, j1, k1), 0));
    new_vel += s1*temp;

    //vec4 new_vel = texture(fieldIn, backtrack).xyzw;
    imageStore(fieldOut, call, new_vel);
}