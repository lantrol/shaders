# Shaders:
In this repository I'll save the programs that I do while I try to learn the use of GLSL.
I will comment the code a bit as notes to help myself and whoever finds them interesting.

## Conway's Game Of Life:
Classic Game Of Life simulation made with Python and GLSL, using PyGame and ModernGL libraries.
Helpful to understand the basics of textures and framebuffers.

![GameOfLifeGIF](https://github.com/user-attachments/assets/e57bca4d-7d63-40b8-a36c-65f44680cd23)

## Boids Simulation:
2D simulation of Boids made using Python Arcade and GLSL, used to learn the basics of compute shaders.
In the following GIF you can see it run on a laptop iGPU with 16384 boids, performing between 50 to 60 FPS.

The algorithm can be found here: https://vanhunteradams.com/Pico/Animal_Movement/Boids-algorithm.html#Separation

![BoidsSimGIF](https://github.com/user-attachments/assets/36af713f-4a60-4bdc-b936-abd0c3b96905)

## Fluid Simulation:
2D fluid simulation made with ModernGL. Thanks to it beeing in 2D, we can simulate it without the need of compute shaders,
by using fragment shaders. Compute shaders will be needed for the 3D version. This simulation is still not "perfect", 
but mostly works as it should. 

The performance is of 45 FPS on a AMD iGPU, simulating a grid of 1024x1024. The simulation its based of "Real-Time Fluid Dynamics for Games" paper.

![FluidSim](https://github.com/user-attachments/assets/d9953744-46dd-483e-8144-8225d789c9c5)

## Fluid Simulation 3D:
3D Fluid simulation made with ModernGL. Uses compute shaders to make easier working with 3D textures. There is also implemented a simple
3D volume rendering inside of a cube, allowing to rotate the cube and see the fluid from multiple angles. The rendering makes the execution slower,
if the rendering code is commented the simulations runs considerably faster.

![fluid3d](https://github.com/user-attachments/assets/0140032a-ba1b-4062-90b4-c8c4d27d8941)

## Numba coded of fluid simulations:
The 2D and 3D simulations are also implemented using Numba for performance comparissons. This code does NOT have realtime rendering, as Numba
doesn't allow this in a easy way.

## Wave Simulations
2D Wave simulation usin FDTDs is made both in plain Python and ModernGL. The use of ModernGL here is a bit different. While in the others the code
is run inside a on_render() loop of the moderngl-window, here OpenGL is in a headless way. No window nor rendering is used, only calling compute shaders
to run when needed. This makes it easier to run shaders from other parts of the code not limited to the on_render() function.
