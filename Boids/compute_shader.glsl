#version 430

layout(local_size_x=GROUPX, local_size_y=GROUPY) in;

struct Boid
{
    vec4 posVel;
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

    float matchFactor = 0.08;

    Boid boid = In.boids[callID];
    Boid out_boid;

    vec4 curInfo = boid.posVel.xyzw;
    vec2 velSum = vec2(0., 0.);

    for (int i = 0; i < In.boids.length(); i++){
        if (i != callID) {
            Boid boid = In.boids[i];
            vec4 otherInfo = boid.posVel.xyzw;
            velSum += otherInfo.zw;
        }
    }

    velSum = velSum/In.boids.length();

    vec2 vel = vec2(0., 0.);
    vel += curInfo.zw+(velSum-curInfo.zw)*matchFactor;
    //vel = vel/(vel.length()+0.00001);
    curInfo.zw = vel;
    curInfo.xy += curInfo.zw;

    out_boid.posVel.xyzw = curInfo.xyzw;
    Out.boids[callID] = out_boid;
}
