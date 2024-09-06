#version 330

in vec2 position;
in float size;

out vec2 vertex_pos;
out float vertex_size;

void main() {
    vertex_pos = position;
    vertex_size = size;
}