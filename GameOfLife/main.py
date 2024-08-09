import os
import sys

import moderngl
import numpy as np
import pygame

SCREEN_SIZE = 800

os.environ['SDL_WINDOWS_DPI_AWARENESS'] = 'permonitorv2'

# Makes Pygame use OpenGL for rendering
pygame.init()
pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

CLOCK = pygame.time.Clock()

class Scene:
    def __init__(self):
        # Detects a context created by the pygame window
        self.ctx = moderngl.create_context()
        # The sampler is used to read textures from within the shaders. Getting the sampler from the context
        # and setting the filter to nearest, we obtain a pixel perfect reading from the texture, needed for the game of life
        self.sampler = self.ctx.sampler(filter=(moderngl.NEAREST, moderngl.NEAREST))
        self.sampler.use()
        # Vertices of a square covering the entire screen. We cant draw anythin withoput a geometry to draw on!
        # First two values are vertex coordenates (range -1, 1), other two are texture coordenates or UVs (range 0, 1)
        # This square covers the entire screen
        vertices = np.array([
            -1.0, -1.0, 0.0, 0.0,
            1.0, -1.0, 1.0, 0.0,
            -1.0, 1.0, 0.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
        ])
        # We create two textures with screensize and 3 components (RGB)
        # The texture can be used by itself, but putting it into a frame buffer allows to use it
        # as a rendering destination
        self.texture0 = self.ctx.texture((SCREEN_SIZE,SCREEN_SIZE), 3)
        self.fb0 = self.ctx.framebuffer(self.texture0)
        self.texture1 = self.ctx.texture((SCREEN_SIZE,SCREEN_SIZE), 3)
        self.fb1 = self.ctx.framebuffer(self.texture1)
        # We write the initial state of the Game of Life, created randomly
        self.texture0.write(self.gen_initial_data(3*SCREEN_SIZE*SCREEN_SIZE).astype(np.uint8).tobytes())
  
        # The vertex shader recieves the information of each vertex defined before, assigning the vertex position
        # to gl_Position, and saving the UVs to be used later. Since the square represents our "Screen", the UVs are
        # similar to our "Screen coordinates".
        # UVs = (X, Y) --> Bottom Left = (0,0) | Bottom Right = (1, 0) | Top Left = (0, 1) | Top Right = (1, 1)
        vert_shader = '''
        #version 330

        in vec2 vert;
        in vec2 texcoord;
        out vec2 uvs;

        void main() {
            uvs = texcoord;
            gl_Position = vec4(vert, 0., 1.);
        }
        '''
        # The fragment shader runs for each pixel of the screen contained in the geometry, in our case the full screen.
        # The fragment shader simply takes a texture (the texture is asigned in the render function)
        # and returns the color of the texture at the UVs position. UVs are received from the vertex shader.
        frag_shader = '''
        #version 330

        uniform sampler2D texture0;
        in vec2 uvs;
        out vec4 f_color;

        void main() {
            f_color = texture(texture0, uvs);
        }
        '''
        # The main program of the Game of Life. Using the texture representing a state of the game, reads
        # the neighboring pixels to calculate the state of itself, returning the color white if alive, or black if dead.
        # texelFetch() is used to read the texture values given a position. The last value is for the level of detail,
        # we leave it at 0
        frag_GOL = '''
        #version 330

        uniform sampler2D texture0;
        in vec2 uvs;
        out vec4 f_color;

        bool alive(vec4 cell) {
            return length(cell.xyz) > 0.1;
        }

        void main() {
            ivec2 pos = ivec2(gl_FragCoord.xy);
            int livingCells = 0;

            vec4 self = texelFetch(texture0, pos, 0);
            vec4 c1 = texelFetch(texture0, pos + ivec2(-1, -1), 0);
            if (alive(c1)) livingCells++;
            vec4 c2 = texelFetch(texture0, pos + ivec2(0, -1), 0);
            if (alive(c2)) livingCells++;
            vec4 c3 = texelFetch(texture0, pos + ivec2(1, -1), 0);
            if (alive(c3)) livingCells++;
            vec4 c4 = texelFetch(texture0, pos + ivec2(-1, 0), 0);
            if (alive(c4)) livingCells++;
            vec4 c5 = texelFetch(texture0, pos + ivec2(1, 0), 0);
            if (alive(c5)) livingCells++;
            vec4 c6 = texelFetch(texture0, pos + ivec2(-1, 1), 0);
            if (alive(c6)) livingCells++;
            vec4 c7 = texelFetch(texture0, pos + ivec2(0, 1), 0);
            if (alive(c7)) livingCells++;
            vec4 c8 = texelFetch(texture0, pos + ivec2(1, 1), 0);
            if (alive(c8)) livingCells++;

            if (alive(self)) {
                if (livingCells == 2 || livingCells == 3) {
                    f_color = vec4(1., 1., 1., 1.);
                } else {
                    f_color = vec4(0., 0., 0., 1.);
                }
            } else {
                if (livingCells == 3) {
                    f_color = vec4(1., 1., 1., 1.);
                } else {
                    f_color = vec4(0., 0., 0., 1.);
                }
            }
        }
        '''
        # We create the programs to be used. The programs define the shaders to be used in sequence
        # The first program will be used to render to the screen the state
        # The second program generates the next state from the current state. It wont render to screen, It will
        # render to another texture.
        self.programRender = self.ctx.program(
            vertex_shader=vert_shader,
            fragment_shader=frag_shader
        )

        self.programLife = self.ctx.program(
            vertex_shader=vert_shader,
            fragment_shader=frag_GOL
        )

        # We load the vertices into a vertex buffer object, allowing to Opengl to read them
        self.vbo = self.ctx.buffer(vertices.astype('f4').tobytes())
        # We create a vertex array object for each program, asignin them the vertex buffer, telling them
        # how to interprete them (two pairs of floats), the first set to the variable "vert" and the second to "texcoord"
        self.vaoRender = self.ctx.vertex_array(self.programRender, [(self.vbo, '2f 2f', 'vert', 'texcoord')])
        self.vaoLife = self.ctx.vertex_array(self.programLife, [(self.vbo, '2f 2f', 'vert', 'texcoord')])

    def gen_initial_data(self, num_values):
        values = np.zeros(num_values)
        for i in range(num_values):
            if np.random.rand() > 0.9:
                values[i] = 255
        return values

    def render(self):
        self.ctx.clear()
        # First we create the next state using the current state saved in texture0.
        # We run the .use() method on both texture0, to access the values of the texture, and fb1, to render the output to texture1
        self.texture0.use()
        self.fb1.use()
        self.vaoLife.render(mode=moderngl.TRIANGLE_STRIP)
        # We render the calculated state in texture1 to the screen
        self.texture1.use()
        self.ctx.screen.use()
        self.vaoRender.render(mode=moderngl.TRIANGLE_STRIP)
        # We use Ping Pong method to reuse the buffers. We swap references in both textures and frame buffers,
        # making the new calculated state the 'current' one for the next iteration
        self.texture0, self.texture1 = self.texture1, self.texture0
        self.fb0, self.fb1 = self.fb1, self.fb0



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
