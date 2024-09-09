import os
import sys

import moderngl
import numpy as np
import pygame

SCREEN_SIZE = 1000
GROUPX = 256
GROUPY = 1

os.environ['SDL_WINDOWS_DPI_AWARENESS'] = 'permonitorv2'

# Makes Pygame use OpenGL for rendering
pygame.init()
pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=False)

CLOCK = pygame.time.Clock()

class Scene:
    def __init__(self):
        # Detects a context created by the pygame window
        self.ctx = moderngl.create_context()
  
        # The vertex shader recieves the information of each vertex defined before, assigning the vertex position
        # to gl_Position, and saving the UVs to be used later. Since the square represents our "Screen", the UVs are
        # similar to our "Screen coordinates".
        # UVs = (X, Y) --> Bottom Left = (0,0) | Bottom Right = (1, 0) | Top Left = (0, 1) | Top Right = (1, 1)
        file = open('vertex_shader.glsl')
        vert_shader = file.read()


        file = open('geometry_shader.glsl')
        geo_shader = file.read()


        # The fragment shader runs for each pixel of the screen contained in the geometry, in our case the full screen.
        # The fragment shader simply takes a texture (the texture is asigned in the render function)
        # and returns the color of the texture at the UVs position. UVs are received from the vertex shader.
        file = open('fragment_shader.glsl')
        frag_shader = file.read()

        # We create the programs to be used. The programs define the shaders to be used in sequence
        # The first program will be used to render to the screen the state
        # The second program generates the next state from the current state. It wont render to screen, It will
        # render to another texture.
        self.program = self.ctx.program(
            vertex_shader=vert_shader,
            geometry_shader=geo_shader,
            fragment_shader=frag_shader
        )

        #X, Y, vel, angle
        num_boids = 2000
        boids_info = []
        for i in range(num_boids):
            a = np.random.rand()-0.5
            b = np.random.rand()-0.5
            c = np.sqrt(b**2 + a**2)/0.006
            boids_info += [np.random.rand()*2-1, np.random.rand()*2-1, a/c, b/c]

        # points = np.array([
        #     0., 0., 0.01, 0.,
        #     0., 0.5, 0.01, 0.006,
        # ])

        points = np.array(boids_info)

        # We load the vertices into a vertex buffer object, allowing to Opengl to read them
        self.vbo1 = self.ctx.buffer(points.astype('f4').tobytes())
        self.vbo2 = self.ctx.buffer(points.astype('f4').tobytes())
        # We create a vertex array object for each program, asignin them the vertex buffer, telling them
        # how to interprete them (two pairs of floats), the first set to the variable "vert" and the second to "texcoord"
        self.vao1 = self.ctx.vertex_array(self.program, [(self.vbo1, '2f 2f', 'position', 'velocity')], mode=self.ctx.POINTS)
        self.vao2 = self.ctx.vertex_array(self.program, [(self.vbo2, '2f 2f', 'position', 'velocity')], mode=self.ctx.POINTS)

        file = open('compute_shader.glsl')
        comp_shader = file.read()
        comp_shader = comp_shader.replace("GROUPX", str(GROUPX))
        comp_shader = comp_shader.replace("GROUPY", str(GROUPY))
        print(comp_shader)

        self.compute_shader = self.ctx.compute_shader(comp_shader)

    def render(self):
        self.ctx.clear()

        self.vbo1.bind_to_storage_buffer(binding=0)
        self.vbo2.bind_to_storage_buffer(binding=1)

        self.compute_shader.run(group_x=GROUPX, group_y=GROUPY)

        self.vao2.render()

        self.vbo1, self.vbo2 = self.vbo2, self.vbo1
        self.vao1, self.vao2 = self.vao2, self.vao1


scene = Scene()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                scene.texture0.write(scene.gen_initial_data(3*SCREEN_SIZE*SCREEN_SIZE).astype(np.uint8).tobytes())
    scene.render()
    pygame.display.flip()
    CLOCK.tick(30)
