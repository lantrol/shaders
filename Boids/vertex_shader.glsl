#version 330

in vec2 position;
in vec2 velocity;

out vec2 vertex_pos;
out vec2 boid_vel;

void main() {
    vertex_pos = position;
    boid_vel = velocity;
}