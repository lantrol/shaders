#version 330

in vec2 g_uv;
in vec3 g_color;

out vec4 f_color;

void main() {
    f_color = vec4(g_color, 1.0);
}