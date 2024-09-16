#version 430

// We define the local group size. The variables are replaced in the python script.
layout(local_size_x=GROUPX, local_size_y=GROUPY) in;

// We define a structure to read the Boids from the input buffer. The memory of the buffer
// is read with a 16 byte alignment, so in case of not using exactly a multiple of 16 bytes,
// padding will be needed
struct Boid
{
    vec4 posVel;
};

// We define a struct-like to read from the buffer binded in the python script
// We will use the buffer binded at 0 as input, the buffer binded at 1 as output
// The buffer struct reads ALL the buffer, we later access by index to our boid of interest
layout(std430, binding=0) buffer boids_in
{
    Boid boids[];
} In;

layout(std430, binding=1) buffer boids_out
{
    Boid boids[];
} Out;

void main() {
    // We obtain the Invocation ID, which we'll use to access a single Boid per compute shader instance
    int callID = int(gl_GlobalInvocationID);

    // Constants used for the simulation
    float matchFactor = MATCH_FACTOR;
    float avoidFactor = AVOID_FACTOR;
    float centeringFactor = CENTERING_FACTOR;
    float turnFactor = TURN_FACTOR;

    float maxSpeed = MAX_SPEED;
    float minSpeed = MIN_SPEED;

    float innerDist = INNER_DIST;
    float outerDist = OUTER_DIST;

    // We read our boid of interest using the invocation id
    Boid in_boid = In.boids[callID];
    // The information is saved in a variable so we can modify it without problems
    vec4 curInfo = in_boid.posVel.xyzw;

    // Variables to save calculations
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
                    // Alignment
                    avgVel += otherInfo.zw;
                    // Centering
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

    // Deffine a boid to save the information, then assign that boid to the output buffer
    Boid out_boid;
    out_boid.posVel.xyzw = curInfo.xyzw;
    Out.boids[callID] = out_boid;
}
