#version 330

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in vec2 vertex_pos[];
in vec2 boid_vel[];

out vec2 g_uv;
out vec3 g_color;

void main(){
    vec2 vPos = vertex_pos[0];
    float vSize = 0.03;
    float vAngle = atan(boid_vel[0].y, boid_vel[0].x);
    mat2 rotate;
    rotate[0] = vec2(cos(vAngle), sin(vAngle));
    rotate[1] = vec2(-sin(vAngle), cos(vAngle));
    g_color = vec3(1., 1., 1.);

    gl_Position = vec4(rotate*vec2(-vSize, vSize/2.) + vPos, 0., 1.);
    g_uv = vec2(0., 1.);
    EmitVertex();

    gl_Position = vec4(rotate*vec2(-vSize/2., 0.) + vPos, 0., 1.);
    g_uv = vec2(0., 0.);
    EmitVertex();

    gl_Position = vec4(rotate*vec2(vSize, 0.) + vPos, 0., 1.);
    g_uv = vec2(1., 1.);
    EmitVertex();

    gl_Position = vec4(rotate*vec2(-vSize, -vSize/2.) + vPos, 0., 1.);
    g_uv = vec2(1., 0.);
    EmitVertex();

    EndPrimitive();
}