# Shaders:
In this repository I'll save the programs that I do while I try to learn the use of GLSL.
I will comment the code a bit as notes to help myself and whoever finds them interesting.

## Conway's Game Of Life:
Classic Game Of Life simulation made with Python and GLSL, using PyGame and ModernGL libraries.
Helpful to understand the basics of textures and framebuffers.

![GameOfLifeGIF](https://github.com/user-attachments/assets/e57bca4d-7d63-40b8-a36c-65f44680cd23)

## Boids Simulation:
2D simulation of Boids made using Python Arcade and GLSL, used to learn the basics of compute shaders.
In the following GIF you can see it run on a laptop iGPU with 10000 Boids at around 30 FPS, totaling 100000000 calculations.
The algorithm can be found here: https://vanhunteradams.com/Pico/Animal_Movement/Boids-algorithm.html#Separation

![BoidsGIF](https://github.com/user-attachments/assets/2475ec5d-2819-407a-ab7a-e06feb4d17bb)
