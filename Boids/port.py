import numpy as np
import arcade
from arcade.gl import BufferDescription

SCREEN_SIZE = 1000

GROUPX = 256
GROUPY = 1

GRAPH_WIDTH = 200
GRAPH_HEIGHT = 120
GRAPH_MARGIN = 5

NUM_BOIDS = 10000
MATCH_FACTOR = 0.03;
AVOID_FACTOR = 0.002;
CENTERING_FACTOR = 0.002;
TURN_FACTOR = 0.0001;

MAX_SPEED = 0.008;
MIN_SPEED = 0.004;

INNER_DIST = 0.035;
OUTER_DIST = 0.1;

class MyWindow(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_SIZE, SCREEN_SIZE,
                                 "Compute Shader",
                                 gl_version=(4, 3),
                                 resizable=True)

        self.center_window()

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
        boids_info = []
        for i in range(NUM_BOIDS):
            a = np.random.rand()-0.5
            b = np.random.rand()-0.5
            c = np.sqrt(b**2 + a**2)/0.006
            boids_info += [np.random.rand()*1.8-0.9, np.random.rand()*1.8-0.9, a/c, b/c]

        # points = np.array([
        #     0., 0., 0.01, 0.,
        #     0., 0.5, 0.01, 0.006,
        # ])

        points = np.array(boids_info)

        # We load the vertices into a vertex buffer object, allowing to Opengl to read them
        self.vbo1 = self.ctx.buffer(data=points.astype('f4').tobytes())
        self.vbo2 = self.ctx.buffer(data=points.astype('f4').tobytes())
        # We create a vertex array object for each program, asignin them the vertex buffer, telling them
        # how to interprete them (two pairs of floats), the first set to the variable "vert" and the second to "texcoord"
        self.vao1 = self.ctx.geometry([BufferDescription(self.vbo1, '2f 2f', ['position', 'velocity'])], mode=self.ctx.POINTS)
        self.vao2 = self.ctx.geometry([BufferDescription(self.vbo2, '2f 2f', ['position', 'velocity'])], mode=self.ctx.POINTS)

        file = open('compute_shader.glsl')
        comp_shader = file.read()
        comp_shader = comp_shader.replace("GROUPX", str(GROUPX))
        comp_shader = comp_shader.replace("GROUPY", str(GROUPY))
        comp_shader = comp_shader.replace("MATCH_FACTOR", str(MATCH_FACTOR))
        comp_shader = comp_shader.replace("TURN_FACTOR", str(TURN_FACTOR))
        comp_shader = comp_shader.replace("AVOID_FACTOR", str(AVOID_FACTOR))
        comp_shader = comp_shader.replace("CENTERING_FACTOR", str(CENTERING_FACTOR))
        comp_shader = comp_shader.replace("MAX_SPEED", str(MAX_SPEED))
        comp_shader = comp_shader.replace("MIN_SPEED", str(MIN_SPEED))
        comp_shader = comp_shader.replace("INNER_DIST", str(INNER_DIST))
        comp_shader = comp_shader.replace("OUTER_DIST", str(OUTER_DIST))
        print(comp_shader)

        self.compute_shader = self.ctx.compute_shader(source=comp_shader)

        # --- Create FPS graph

        # Enable timings for the performance graph
        arcade.enable_timings()

        # Create a sprite list to put the performance graph into
        self.perf_graph_list = arcade.SpriteList()

        # Create the FPS performance graph
        graph = arcade.PerfGraph(GRAPH_WIDTH, GRAPH_HEIGHT, graph_data="FPS")
        graph.center_x = GRAPH_WIDTH / 2
        graph.center_y = self.height - GRAPH_HEIGHT / 2
        self.perf_graph_list.append(graph)

    def on_draw(self):
        self.clear()

        # Enable blending so our alpha channel works
        self.ctx.enable(self.ctx.BLEND)

        self.vbo1.bind_to_storage_buffer(binding=0)
        self.vbo2.bind_to_storage_buffer(binding=1)

        self.compute_shader.run(group_x=GROUPX, group_y=GROUPY)

        self.vao2.render(self.program)

        self.vbo1, self.vbo2 = self.vbo2, self.vbo1
        self.vao1, self.vao2 = self.vao2, self.vao1

        # Draw the graphs
        self.perf_graph_list.draw()

app = MyWindow()
arcade.run()
