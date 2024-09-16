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
        # to gl_Position, and saving the UVs. In this case UVs are not used.
        # UVs = (X, Y) --> Bottom Left = (0,0) | Bottom Right = (1, 0) | Top Left = (0, 1) | Top Right = (1, 1)
        file = open('vertex_shader.glsl')
        vert_shader = file.read()

        # For each vertex, it will generate 4 new vertex with the shape of an arrow. Each arrow represents a boid
        file = open('geometry_shader.glsl')
        geo_shader = file.read()


        # The fragmen shader draws the boids white.
        file = open('fragment_shader.glsl')
        frag_shader = file.read()

        # We create the programs to be used. The programs define the shaders to be used in sequence
        # The vertex info recieved by the vertex shader is passed to de geometry shader. The geometry shader
        # creates a quad for each point received.
        # The fragment shader draws white in the geometry of each Boid.
        self.program = self.ctx.program(
            vertex_shader=vert_shader,
            geometry_shader=geo_shader,
            fragment_shader=frag_shader
        )

        #We generate the attributes of each boid. Each boid has 4 floats: (Xpos, Ypos, Xvel, Yvel)
        boids_info = []
        for i in range(NUM_BOIDS):
            a = np.random.rand()-0.5
            b = np.random.rand()-0.5
            c = np.sqrt(b**2 + a**2)/0.006
            boids_info += [np.random.rand()*1.8-0.9, np.random.rand()*1.8-0.9, a/c, b/c]

        points = np.array(boids_info)

        # We load the vertices into a vertex buffer object, allowing to Opengl to read them
        self.vbo1 = self.ctx.buffer(data=points.astype('f4').tobytes())
        self.vbo2 = self.ctx.buffer(reserve=self.vbo1.size)
        # We create two vertex arrays, one used to read the current Boids state, the other to write the next state into.
        # Using the BufferDescription, we tell how to interprete the buffers. In this case, we tell which buffer to read from,
        # to read two pairs of floats (each pair is a vec2 type), and the name of each variable.
        # (position = vec2 and velocity = vec2)
        self.vao1 = self.ctx.geometry([BufferDescription(self.vbo1, '2f 2f', ['position', 'velocity'])], mode=self.ctx.POINTS)
        self.vao2 = self.ctx.geometry([BufferDescription(self.vbo2, '2f 2f', ['position', 'velocity'])], mode=self.ctx.POINTS)

        # Whe load the compute shader and replace all the constants with the ones defined in the python script.
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

        # We define a compute shader and give it the file as source
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

        # We bind each vertex buffer to a different channel, making the accesible from the compute shader
        self.vbo1.bind_to_storage_buffer(binding=0)
        self.vbo2.bind_to_storage_buffer(binding=1)

        # We run the compute shader with the defined WorkGroup size.
        # Don't quite understand it yet, I leave it a (256, 1)
        # For each 4 floats representing a Boid in the buffer, a instance of the compute shader will be run.
        self.compute_shader.run(group_x=GROUPX, group_y=GROUPY)

        # We render the calculated Boids at the second vertex array. We give it the program used to render.
        self.vao2.render(self.program)

        # We ping pong the vertex arrays and buffers, using the same memory to calculate the next state
        self.vbo1, self.vbo2 = self.vbo2, self.vbo1
        self.vao1, self.vao2 = self.vao2, self.vao1

        # Draw the graphs
        self.perf_graph_list.draw()

app = MyWindow()
arcade.run()
