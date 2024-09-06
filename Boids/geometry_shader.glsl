#version 330

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in vec2 vertex_pos[];

out vec2 g_uv;
out vec3 g_color;

void main(){
    vec2 vPos = vertex_pos[0];
    float vSize = 0.1;
    float angle = 1.;
    mat2 rotate;
    rotate[0] = vec2(cos(angle), sin(angle));
    rotate[1] = vec2(-sin(angle), cos(angle));
    g_color = vec3(1., 1., 1.);

    gl_Position = vec4(rotate*vec2(-vSize, vSize) + vPos, 0., 1.);
    g_uv = vec2(0, 1);
    EmitVertex();

    gl_Position = vec4(rotate*vec2(-vSize, -vSize) + vPos, 0., 1.);
    g_uv = vec2(0, 0);
    EmitVertex();

    gl_Position = vec4(rotate*vec2(vSize, vSize) + vPos, 0., 1.);
    g_uv = vec2(1, 1);
    EmitVertex();

    gl_Position = vec4(rotate*vec2(vSize, -vSize) + vPos, 0., 1.);
    g_uv = vec2(1, 0);
    EmitVertex();

    EndPrimitive();
}