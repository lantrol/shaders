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

    float matchFactor = MATCH_FACTOR;
    float avoidFactor = AVOID_FACTOR;
    float centeringFactor = CENTERING_FACTOR;
    float turnFactor = TURN_FACTOR;

    float maxSpeed = MAX_SPEED;
    float minSpeed = MIN_SPEED;

    float innerDist = INNER_DIST;
    float outerDist = OUTER_DIST;

    Boid in_boid = In.boids[callID];
    Boid out_boid;

    vec4 curInfo = in_boid.posVel.xyzw;


    vec2 avgVel = vec2(0., 0.);
    vec2 closeVel = vec2(0., 0.);
    vec2 centerPos = vec2(0., 0.);
    int neighborCount = 0;
    for (int i = 0; i < In.boids.length(); i++){
        if (i != callID) {
            Boid otherBoid = In.boids[i];
            vec4 otherInfo = otherBoid.posVel.xyzw;

            vec2 betweenVec = curInfo.xy - otherInfo.xy;
            float dist = length(betweenVec);
            if (dist < outerDist){
                if (dist < innerDist) {
                    // Separation
                    closeVel += betweenVec;
                }
                else {
                    avgVel += otherInfo.zw;
                    centerPos += otherInfo.xy;
                    neighborCount++;
                }
            }
        }
    }

    if (neighborCount > 0){
        avgVel = avgVel/neighborCount;
        centerPos = centerPos/neighborCount;
    }

    //vel = vel/(length(vel)+0.00001);
    curInfo.zw += (avgVel.xy - curInfo.zw)*matchFactor;
    curInfo.zw += closeVel.xy*avoidFactor;
    curInfo.zw += (centerPos - curInfo.xy)*centeringFactor;

    // Screen edge
    if (curInfo.x < -0.95)
        curInfo.z += turnFactor;
    if (curInfo.x > 0.95)
        curInfo.z -= turnFactor;
    if (curInfo.y < -0.95)
        curInfo.w += turnFactor;
    if (curInfo.y > 0.95)
        curInfo.w -= turnFactor;

    float speed = length(curInfo.zw);
    if (speed > maxSpeed) {
        curInfo.zw = (curInfo.zw/speed)*maxSpeed;
    }
    else if (speed < minSpeed) {
        curInfo.zw = (curInfo.zw/speed)*minSpeed;
    }

    curInfo.xy += curInfo.zw;

    out_boid.posVel.xyzw = curInfo.xyzw;
    Out.boids[callID] = out_boid;
}
