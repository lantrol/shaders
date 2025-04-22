import numpy as np
import moderngl
import moderngl_window as mglw
from moderngl_window import geometry
from moderngl_window.geometry.attributes import AttributeNames
import time as clock

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800

GRID_WIDTH = 64
GRID_HEIGHT = 64
GRID_DEPTH = 64

GROUPX = 8
GROUPY = 8
GROUPZ = 8

DT = 1/60

vertices = np.array([
    -1.0, -1.0, 0., 0.,
    -1.0, 1.0, 0., 1.,
    1.0, -1.0, 1., 0.,
    -1.0, 1.0, 0., 1.,
    1.0, -1.0, 1., 0.,
    1.0, 1.0, 1., 1.,
])

vert_shader = '''
#version 430 core

in vec2 vert;
in vec2 texcoord;
out vec2 uvs;

void main() {
    uvs = texcoord;
    gl_Position = vec4(vert, 0., 1.);
}
'''

frag_shader = '''
#version 430 core

in vec2 uvs;
uniform sampler3D textura;
out vec4 f_color;

void main() {
    //float vel_x = abs(texture(textura, vec3(uvs.xy, 0.5)).x*20.);
    //float vel_y = abs(texture(textura, vec3(uvs.xy, 0.5)).y*2.);
    //float vel_z = abs(texture(textura, vec3(uvs.xy, 0.5)).z*2.);
    //float vel_x = abs(texture(textura, vec3(uvs.x, 0.5, uvs.y)).x*10.);
    //float vel_y = abs(texture(textura, vec3(uvs.x, 0.5, uvs.y)).y*10.);
    //float vel_z = abs(texture(textura, vec3(uvs.x, 0.5, uvs.y)).z*10.);
    float dens = texture(textura, vec3(uvs.xy, 0.5)).w*10.;
    //f_color = vec4(vel_x, vel_x, vel_x, 1.0);
    //float dens = log(texture(textura, vec3(uvs.xy, 0.5)).y+1)*100000;
    f_color = vec4(dens, dens, dens, 1.0);
}
'''

simple_volume_shader = '''
#version 430 core

in vec2 uvs;
uniform sampler3D textura;
out vec4 f_color;

void main() {
    float depth = 0.0;
    float dens = 0.0;
    while (depth < 1.0) {
        //float vel_x = abs(texture(textura, vec3(uvs.xy, 0.5)).x*10.);
        //float vel_y = abs(texture(textura, vec3(uvs.xy, 0.5)).y*10.);
        //float vel_z = abs(texture(textura, vec3(uvs.xy, 0.5)).z*10.);
        dens += texture(textura, vec3(uvs.xy, depth)).w*10;
        //dens += log(texture(textura, vec3(uvs.xy, depth)).y+1)*6;
        depth += 0.1;
    }
    f_color = clamp(vec4(dens, dens, dens, 1.0), 0., 1.);
}
'''


# External conditions:
#       - x: Force in X axis
#       - y: Force in Y axis
#       - z: Color input
#
# Field texture:
#       - x: Velocity in X axis
#       - y: Velocity in Y axis
#       - z: Color density
#
# PDiv texture:
#       - x: Divergence
#       - y: p field

class TestWindow(mglw.WindowConfig):
    gl_version = (4,3)
    window_size = (SCREEN_WIDTH, SCREEN_HEIGHT)
    aspect_ratio = SCREEN_WIDTH/SCREEN_HEIGHT
    vsync = False

    def __init__(self, ctx = None, wnd = None, timer = None, **kwargs):
        super().__init__(ctx, wnd, timer, **kwargs)
        self.forces = self.ctx.texture3d((GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH), 4, dtype="f4")
        self.field0 = self.ctx.texture3d((GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH), 4, dtype="f4")
        self.field1 = self.ctx.texture3d((GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH), 4, dtype="f4")
        self.pdiv0 = self.ctx.texture3d((GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH), 4, dtype="f4")
        self.pdiv1 = self.ctx.texture3d((GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH), 4, dtype="f4")
        self.quad = geometry.quad_fs(AttributeNames("vert", texcoord_0="texcoord"), normals=False)
        self.renderer = self.ctx.program(vertex_shader=vert_shader, fragment_shader=frag_shader)
        self.add_forces = self.load_compute_shader(
            "C:\\Users\\jimenez.147235\\Desktop\\shaders\\FluidSim3D\\add_forces_compute.glsl"
        )
        self.advect = self.load_compute_shader(
            "C:\\Users\\jimenez.147235\\Desktop\\shaders\\FluidSim3D\\advect_manual_compute.glsl"
        )
        self.project_divergence = self.load_compute_shader(
            "C:\\Users\\jimenez.147235\\Desktop\\shaders\\FluidSim3D\\divergence_boundary_compute.glsl"
        )
        self.project_pressure = self.load_compute_shader(
            "C:\\Users\\jimenez.147235\\Desktop\\shaders\\FluidSim3D\\pressure_boundary_compute.glsl"
        )
        self.project_velocity = self.load_compute_shader(
            "C:\\Users\\jimenez.147235\\Desktop\\shaders\\FluidSim3D\\velocity_boundary_compute.glsl"
        )
        self.create_texture()
        self.iters = 1
        # self.sampler = self.ctx.sampler(False, False, False, (moderngl.LINEAR, moderngl.LINEAR))
        # self.sampler.use()

        self.add_forces["GRID_WIDTH"] = GRID_WIDTH
        self.add_forces["GRID_HEIGHT"] = GRID_HEIGHT
        self.add_forces["GRID_DEPTH"] = GRID_DEPTH
        self.add_forces["DT"] = DT

        self.advect["GRID_WIDTH"] = GRID_WIDTH
        self.advect["GRID_HEIGHT"] = GRID_HEIGHT
        self.advect["GRID_DEPTH"] = GRID_DEPTH
        self.advect["DT"] = DT

        self.project_divergence["GRID_WIDTH"] = GRID_WIDTH
        self.project_divergence["GRID_HEIGHT"] = GRID_HEIGHT
        self.project_divergence["GRID_DEPTH"] = GRID_DEPTH

        self.project_pressure["GRID_WIDTH"] = GRID_WIDTH
        self.project_pressure["GRID_HEIGHT"] = GRID_HEIGHT
        self.project_pressure["GRID_DEPTH"] = GRID_DEPTH

        self.project_velocity["GRID_WIDTH"] = GRID_WIDTH
        self.project_velocity["GRID_HEIGHT"] = GRID_HEIGHT
        self.project_velocity["GRID_DEPTH"] = GRID_DEPTH

        self.start = clock.time()

    def on_render(self, time, frame_time):
        #return super().render(time, frame_time)
        self.ctx.clear()

        if self.iters % 3900 == 0:
            end = clock.time()
            print("Elapsed time, second run: {} || Frames per second: {}".format((end-self.start), 3900/(end-self.start)))
            self.start = clock.time()

        # Add Forces
        self.forces.use(0)
        self.field0.use(1)
        self.field1.bind_to_image(2, read=False, write=True)
        self.add_forces.run(group_x = GROUPX, group_y = GROUPY, group_z=GROUPZ)
        self.swap_fields()

        # Advect
        self.field0.use(0)
        self.field1.bind_to_image(1, read=False, write=True)
        self.advect.run(group_x = GROUPX, group_y = GROUPY, group_z=GROUPZ)
        self.swap_fields()

        # Projection
        # - Divergence
        self.field0.use(0)
        self.pdiv0.bind_to_image(1, read=False, write=True)
        self.project_divergence.run(group_x = GROUPX, group_y = GROUPY, group_z=GROUPZ)

        # - Pressure
        for k in range(20):
            self.pdiv0.use(0)
            self.pdiv1.bind_to_image(1, read=False, write=True)
            self.project_pressure.run(group_x = GROUPX, group_y = GROUPY, group_z=GROUPZ)
            self.pdiv0, self.pdiv1 = self.pdiv1, self.pdiv0

        # - Velocity
        self.pdiv0.use(0)
        self.field0.use(1)
        self.field1.bind_to_image(2, read=False, write=True)
        self.project_velocity.run(group_x = GROUPX, group_y = GROUPY, group_z=GROUPZ)
        self.swap_fields()

        self.iters += 1

        # Render
        self.field0.use(0)
        #self.pdiv0.use(0)
        self.ctx.screen.use()
        self.quad.render(self.renderer)

        


    def swap_fields(self):
        self.field0, self.field1 = self.field1, self.field0


    def create_texture(self):
        tex = np.zeros([self.field0.size[0], self.field0.size[1], self.field0.size[2], 4]).astype(np.float32)

        # for i in range(1, GRID_WIDTH+1):
        #     for j in range(1, GRID_HEIGHT+1):
        #         for k in range(1, GRID_DEPTH+1):
        #             tex[i, j, k, 0] = (np.random.randint(100)-50)/500
        #             tex[i, j, k, 1] = (np.random.randint(100)-50)/500
        #             tex[i, j, k, 2] = (np.random.randint(100)-50)/500
        
        self.field0.write(tex.ravel().tobytes())
        self.field1.write(tex.ravel().tobytes())
        tex2 = np.zeros([self.field0.size[0], self.field0.size[1], self.field0.size[2], 4]).astype(np.float32)
        # tex2[:, :, :, :] = 1.
        mid, width = (GRID_WIDTH//2-2), 4
        
        #tex2[mid+1:mid+width+1, 1:4, mid+1:mid+width+1, 1] = 0.1
        #tex2[mid+1:mid+width+1, 1:4, mid+1:mid+width+1, 3] = 0.1
        #tex2[mid+1:mid+width+1, -4:-1, mid+1:mid+width+1, 1] = -0.1
        #tex2[mid+1:mid+width+1, -4:-1, mid+1:mid+width+1, 3] = 0.1

        # tex2[:width, 1:4, mid:mid+width, 1] = 1
        # tex2[:width, 1:4, mid:mid+width, 2] = 1
        # tex2[:width, 1:4, mid:mid+width, 3] = 0.5
        # tex2[GRID_DEPTH-width+1:GRID_DEPTH+1, -4:-1, mid:mid+width, 1] = -1
        # tex2[GRID_DEPTH-width+1:GRID_DEPTH+1, -4:-1, mid:mid+width, 2] = -1
        # tex2[GRID_DEPTH-width+1:GRID_DEPTH+1, -4:-1, mid:mid+width, 3] = 0.5

        # tex2[mid//2:width//2+mid, mid//2:mid//2+1, mid:mid+width, 1] = 1
        # tex2[mid//2:width//2+mid, mid//2:mid//2+1, mid:mid+width, 3] = 0.5

        tex2[GRID_DEPTH//2-1:GRID_DEPTH//2, 9:10, GRID_DEPTH//2-1:GRID_DEPTH//2, 1] = 0.5
        tex2[GRID_DEPTH//2-1:GRID_DEPTH//2, GRID_DEPTH-10-1:GRID_DEPTH-9-1, GRID_DEPTH//2-1:GRID_DEPTH//2, 1] = -0.5
        tex2[GRID_DEPTH//2-1:GRID_DEPTH//2, 9:10, GRID_DEPTH//2-1:GRID_DEPTH//2, 3] = 0.5
        tex2[GRID_DEPTH//2-1:GRID_DEPTH//2, GRID_DEPTH-10-1:GRID_DEPTH-9-1, GRID_DEPTH//2-1:GRID_DEPTH//2, 3] = 0.5
        # tex2[mid:width+mid, 26:27, mid:mid+width, 1] = 1
        # tex2[mid:width+mid, 26:27, mid:mid+width, 3] = 0.5

        self.forces.write(tex2.ravel().tobytes())


mglw.run_window_config(TestWindow)
