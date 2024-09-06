#version 430

layout(local_size_x=256, local_size_y=1) in;

struct Boid
{
    vec4 info;
};

layout(std430, binding=0) buffer boids_in
{
    Boid boids[];
} In;

layout(std430, binding=1) buffer boids_out
{
    Boid boids[];
} Out;

void main() {
    int callID = int(gl_GlobalInvocationID);

    Boid in_boid = In.boids[callID];
    Boid out_boid;

    vec4 curInfo = in_boid.info.xyzw;
    curInfo.x = curInfo.x + 0.01;

    out_boid.info.xyzw = curInfo.xyzw;
    Out.boids[callID] = out_boid;
}