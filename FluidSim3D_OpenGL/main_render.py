import numpy as np
import moderngl
import moderngl_window as mglw
from moderngl_window import geometry
from moderngl_window.geometry.attributes import AttributeNames
import glm
import time as clock
from math import sin

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 1200

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

vert_volume = '''
#version 430 core

in vec3 vert;
in vec3 normals;
in vec2 texcoord;
out vec2 uvs;
out vec3 vert_pos;
out vec3 t_vert_pos;

uniform mat4 position;
uniform mat4 rotation;
uniform mat4 view;
uniform mat4 projection;

void main() {
    uvs = texcoord;
    vert_pos = vert;
    t_vert_pos = (projection*view*position*rotation*vec4(vert, 1.)).xyz;
    gl_Position = projection*view*position*rotation*vec4(vert, 1.);
}
'''

frag_volume = '''
#version 430 core

in vec2 uvs;
in vec3 vert_pos;
in vec3 t_vert_pos;
out vec4 f_color;

uniform sampler3D volume;
uniform mat4 position;
uniform mat4 rotation;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 camera_pos;
uniform float step_len = 0.01;

void main() {
    if (uvs.x < 0.01 || uvs.x > 0.99 || uvs.y < 0.01 || uvs.y > 0.99) {
        f_color = vec4(0.33, 0.33, 0.33, 1.);
        return;
    }
    vec3 ray_point = (inverse(projection*view*position*rotation)*vec4(camera_pos, 1.)).xyz;
    vec3 dir = normalize(vert_pos - ray_point);
    float dens = 0.0;
    float color = 0.0;
    float distance = 0.0;
    vec3 pos = vert_pos + dir*step_len;
    float leng = step_len;
    while (abs(pos.x) < 0.5 && abs(pos.y) < 0.5 && abs(pos.z) < 0.5 && dens < 1.0) {
        float pos_dens = 2*texture(volume, pos+vec3(0.5, 0.5, 0.5)).w;
        dens += pos_dens*step_len;

        // Calculate blocked light
        vec3 block_pos = pos;
        vec3 light_dir = vec3(0., 1., 0.);
        float light_len = 0.05;
        float block = 0.0;
        while (abs(block_pos.x) < 0.5 && abs(block_pos.y) < 0.5 && abs(block_pos.z) < 0.5) {
            block += 20.*texture(volume, block_pos+vec3(0.5, 0.5, 0.5)).w*0.05;
            block_pos += 0.05*light_dir;
        }

        pos += dir*step_len;
        leng += step_len;
        color += exp(-(block+dens)*2.)*pos_dens;
        color = clamp(color, 0.001, 0.999);
    }
    f_color = vec4(color, color, color, 1.-exp(-dens*10.));
}
'''

frag_volume_simple = '''
#version 430 core

in vec2 uvs;
in vec3 vert_pos;
in vec3 t_vert_pos;
out vec4 f_color;

uniform sampler3D volume;
uniform mat4 position;
uniform mat4 rotation;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 camera_pos;
uniform float step_len = 0.01;

void main() {
    if (uvs.x < 0.01 || uvs.x > 0.99 || uvs.y < 0.01 || uvs.y > 0.99) {
        f_color = vec4(0.33, 0.33, 0.33, 1.);
        return;
    }
    vec3 ray_point = (inverse(projection*view*position*rotation)*vec4(camera_pos, 1.)).xyz;
    vec3 dir = normalize(vert_pos - ray_point);
    float dens = 0.0;
    vec3 pos = vert_pos + dir*step_len;
    while (abs(pos.x) < 0.5 && abs(pos.y) < 0.5 && abs(pos.z) < 0.5 && dens < 1.0) {
        float pos_dens = step_len*texture(volume, pos+vec3(0.5, 0.5, 0.5)).w;
        dens += pos_dens;
        pos += dir*step_len;
    }
    f_color = vec4(1.0, 1.0, 1.0, 1-exp(-dens*10.));
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
        self.renderer = self.ctx.program(vertex_shader=vert_volume, fragment_shader=frag_volume_simple)
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

        # Geometry setup
        self.cube = geometry.cube(attr_names=AttributeNames("vert", normal="normals", texcoord_0="texcoord"), normals=True)
        self.rotation_x = 0.0
        self.rotation_y = 0.0

        # Uniforms Setup
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
        self.ctx.clear(0.2, 0.2, 0.2, 1.0)
        self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)

        if self.iters % 2000 == 0:
            end = clock.time()
            print("Elapsed time, second run: {} || Frames per second: {}".format((end-self.start), 2000/(end-self.start)))
            print(time)
            self.start = clock.time()

        # Add Forces
        self.forces.use(0)
        self.field0.use(1)
        self.field1.bind_to_image(2, read=False, write=True)
        #self.add_forces["u_time"] = time
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

        #Render
        rotation = glm.rotate(glm.mat4(1.0), glm.radians(self.rotation_y), glm.vec3(1., 0., 0.))
        rotation = glm.rotate(rotation, glm.radians(self.rotation_x), glm.vec3(0., 1., 0.))
        position = glm.translate(glm.mat4(1.0), glm.vec3(0, 0, 0))
        camera_pos = glm.vec3(0, 0, -3)
        view = glm.mat4(1.0)
        view = glm.translate(view, camera_pos)
        projection = glm.mat4()
        projection = glm.ortho(-1., 1., -1., 1., 0., 100.)

        self.renderer["position"].write(position)
        self.renderer["rotation"].write(rotation)
        self.renderer["view"].write(view)
        self.renderer["projection"].write(projection)
        self.renderer["camera_pos"].write(camera_pos)
        self.renderer["volume"] = 0
        self.field0.use(0)
        self.ctx.screen.use()
        self.cube.render(self.renderer)


    def on_mouse_drag_event(self, x, y, dx, dy):
        #print("Rotations:", self.rotation_x, self.rotation_y)
        self.rotation_x += dx
        self.rotation_y += dy


    def swap_fields(self):
        self.field0, self.field1 = self.field1, self.field0


    def create_texture(self):
        tex = np.zeros([self.field0.size[0], self.field0.size[1], self.field0.size[2], 4]).astype(np.float32)

        #for i in range(1, GRID_WIDTH):
        #    for j in range(1, GRID_HEIGHT):
        #        for k in range(1, GRID_DEPTH):
        #            tex[i, j, k, 0] = ((np.random.randint(100)-50)/200)**2
        #            tex[i, j, k, 1] = ((np.random.randint(100)-50)/200)**2
        #            tex[i, j, k, 2] = ((np.random.randint(100)-50)/200)**2

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

        # tex2[GRID_DEPTH//2-1:GRID_DEPTH//2+1, 9:10, GRID_DEPTH//2-1:GRID_DEPTH//2+1, 1] = 0.5
        # tex2[GRID_DEPTH//2-1:GRID_DEPTH//2+1, GRID_DEPTH-10-1:GRID_DEPTH-9-1, GRID_DEPTH//2-1:GRID_DEPTH//2+1, 1] = -0.5
        # tex2[GRID_DEPTH//2-1:GRID_DEPTH//2+1, 9:10, GRID_DEPTH//2-1:GRID_DEPTH//2+1, 3] = 1
        # tex2[GRID_DEPTH//2-1:GRID_DEPTH//2+1, GRID_DEPTH-10-1:GRID_DEPTH-9-1, GRID_DEPTH//2-1:GRID_DEPTH//2+1, 3] = 1

        for i in range(1, GRID_WIDTH):
            for j in range(1, GRID_HEIGHT):
                value = max(abs(sin(i/2)) + sin(i/2) + abs(sin(j/2)) + sin(j/2), 0)**2
                value = value if value > 10 else 0
                tex2[i, 2:4, j, 1] = value/64
                tex2[i, GRID_HEIGHT-4:GRID_HEIGHT-2, j, 1] = -value/64
                tex2[i, 2:4, j, 3] = value/128
                tex2[i, GRID_HEIGHT-4:GRID_HEIGHT-2, j, 3] = value/128

        # tex2[GRID_DEPTH//2-4:GRID_DEPTH//2, 4:8, GRID_DEPTH//2-4:GRID_DEPTH//2, 1] = 1
        # tex2[GRID_DEPTH//2-4:GRID_DEPTH//2, 4:8, GRID_DEPTH//2-4:GRID_DEPTH//2, 3] = 1
        # tex2[GRID_DEPTH//2-2:GRID_DEPTH//2, GRID_DEPTH//2-2:GRID_DEPTH//2, 4:6, 0] = 1
        # tex2[GRID_DEPTH//2-2:GRID_DEPTH//2, GRID_DEPTH//2-2:GRID_DEPTH//2, 4:6, 3] = 0.5

        self.forces.write(tex2.ravel().tobytes())


mglw.run_window_config(TestWindow)
