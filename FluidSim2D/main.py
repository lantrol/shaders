import numpy as np
import arcade
import moderngl
import moderngl_window as mglw
from moderngl_window import geometry
from moderngl_window.geometry.attributes import AttributeNames

SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 1024

GROUPX = 128
GROUPY = 1

LOCALX = 125
LOCALY = 40

vertices = np.array([
    -1.0, -1.0, 0., 0.,
    -1.0, 1.0, 0., 1.,
    1.0, -1.0, 1., 0.,
    -1.0, 1.0, 0., 1.,
    1.0, -1.0, 1., 0.,
    1.0, 1.0, 1., 1.,
])

vert_shader = '''
#version 330 core

in vec2 vert;
in vec2 texcoord;
out vec2 uvs;

void main() {
    uvs = texcoord;
    gl_Position = vec4(vert, 0., 1.);
}
'''

frag_shader = '''
#version 330 core

in vec2 uvs;
uniform sampler2D textura;
out vec4 f_color;

void main() {
    //f_color = vec4(vec3(min(texture(textura, uvs).z, 1.)), 1.0);
    f_color = vec4(abs(texture(textura, uvs).z)*1., abs(texture(textura, uvs).z)*1., 0., 1.0);
}
'''

add_forces_shader = '''
#version 330 core

in vec2 uvs;
uniform sampler2D field;
uniform sampler2D forces;
out vec4 f_color;

void main() {
    vec3 new_dens = texture(field, uvs).xyz + texture(forces, uvs).xyz;
    f_color = vec4(new_dens, 1.);
}
'''

spread_shader ='''
#version 330 core

in vec2 uvs;
uniform sampler2D textura;
uniform float dt;
uniform int WIDTH;
out vec4 f_color;

void main() {
    //f_color = vec4(1., 1., 1., 1.);
    float diff_factor = 0.2;
    float a = dt*diff_factor*WIDTH*WIDTH;
    ivec2 pos = ivec2(gl_FragCoord.xy);
    vec4 center = texture(textura, uvs);
    vec4 top = texelFetch(textura, pos + ivec2(0, 1), 0);
    vec4 down = texelFetch(textura, pos + ivec2(0, -1), 0);
    vec4 left = texelFetch(textura, pos + ivec2(-1, 0), 0);
    vec4 right = texelFetch(textura, pos + ivec2(1, 0), 0);
    f_color = vec4((center.xyz + a*(top.xyz + down.xyz + left.xyz + right.xyz)) / (1. + 4.*a), 1.);
}
'''

advect_shader ='''
#version 330 core

in vec2 uvs;
uniform sampler2D textura;
uniform float dt;
uniform int imageSize;
out vec4 f_color;

void main() {
    ivec2 curPos = ivec2(gl_FragCoord.xy);
    if (min(curPos.x, curPos.y) != 0 && max(curPos.x, curPos.y) != imageSize) {
        vec2 vel = texture(textura, uvs).xy;
        vec2 backtrack = uvs - dt*vel;
        f_color = vec4(texture(textura, backtrack).xyz, 1.0);
    }
    else {
        f_color = vec4(texelFetch(textura, curPos, 0).xyz, 1.0);
    }
}
'''

divergence_shader = '''
#version 330 core

in vec2 uvs;
uniform sampler2D textura;
uniform int imageSize;
out vec4 f_color;

void main() {
    float h = 1.0/800.;
    ivec2 pos = ivec2(gl_FragCoord.xy);
    if (min(pos.x, pos.y) != 0 && max(pos.x, pos.y) != imageSize) {
        float div_value = texelFetch(textura, pos + ivec2(1, 0), 0).x - texelFetch(textura, pos - ivec2(1, 0), 0).x;
        div_value = div_value + texelFetch(textura, pos + ivec2(0, 1), 0).y - texelFetch(textura, pos - ivec2(0, 1), 0).y;
        div_value = -0.5*h*div_value;
        f_color = vec4(div_value, 0.0, 0.0, 0.0);
    }
}
'''

p_shader = '''
#version 330 core

in vec2 uvs;
uniform sampler2D textura;
out vec4 f_color;

void main() {
    ivec2 pos = ivec2(gl_FragCoord.xy);
    vec4 pos_value = texelFetch(textura, pos, 0);
    float p_value = pos_value.x;
    p_value = p_value + texelFetch(textura, pos + ivec2(1, 0), 0).y + texelFetch(textura, pos - ivec2(1, 0), 0).y;
    p_value = p_value + texelFetch(textura, pos + ivec2(0, 1), 0).y + texelFetch(textura, pos - ivec2(0, 1), 0).y;
    p_value = p_value/4.0;
    f_color = vec4(pos_value.x, p_value, pos_value.z, pos_value.w);
}
'''

last_vel_shader = '''
#version 330 core

in vec2 uvs;
uniform sampler2D textura;
uniform sampler2D field;
out vec4 f_color;

void main() {
    float h = 1.0/800;
    ivec2 pos = ivec2(gl_FragCoord.xy);
    vec4 pos_value = texelFetch(field, pos, 0);
    float new_x = pos_value.x - 0.5*(texelFetch(textura, pos + ivec2(1, 0), 0).y - texelFetch(textura, pos - ivec2(1, 0), 0).y)/h;
    float new_y = pos_value.y - 0.5*(texelFetch(textura, pos + ivec2(0, 1), 0).y - texelFetch(textura, pos - ivec2(0, 1), 0).y)/h;
    f_color = vec4(new_x, new_y, texelFetch(field, pos, 0).zw);
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
    gl_version = (3,3)
    window_size = (SCREEN_WIDTH, SCREEN_HEIGHT)
    aspect_ratio = SCREEN_WIDTH/SCREEN_HEIGHT

    def __init__(self, ctx = None, wnd = None, timer = None, **kwargs):
        super().__init__(ctx, wnd, timer, **kwargs)
        self.external_conditions = self.ctx.texture((SCREEN_WIDTH, SCREEN_WIDTH), 4, dtype="f4")
        self.field0 = self.ctx.texture((SCREEN_HEIGHT, SCREEN_WIDTH), 4, dtype="f4")
        self.field1 = self.ctx.texture((SCREEN_HEIGHT, SCREEN_WIDTH), 4, dtype="f4")
        self.pdiv0 = self.ctx.texture((SCREEN_HEIGHT, SCREEN_WIDTH), 4, dtype="f4")
        self.pdiv1 = self.ctx.texture((SCREEN_HEIGHT, SCREEN_WIDTH), 4, dtype="f4")
        self.fboField0 = self.ctx.framebuffer(self.field0)
        self.fboField1 = self.ctx.framebuffer(self.field1)
        self.fboPdiv0 = self.ctx.framebuffer(self.pdiv0)
        self.fboPdiv1 = self.ctx.framebuffer(self.pdiv1)
        self.quad = geometry.quad_fs(AttributeNames("vert", texcoord_0="texcoord"), normals=False)
        self.program = self.ctx.program(vertex_shader=vert_shader, fragment_shader=frag_shader)
        self.spread = self.ctx.program(vertex_shader=vert_shader, fragment_shader=spread_shader)
        self.advect = self.ctx.program(vertex_shader=vert_shader, fragment_shader=advect_shader)
        self.divergence = self.ctx.program(vertex_shader=vert_shader, fragment_shader=divergence_shader)
        self.p_calc = self.ctx.program(vertex_shader=vert_shader, fragment_shader=p_shader)
        self.last_prog = self.ctx.program(vertex_shader=vert_shader, fragment_shader=last_vel_shader)
        self.add_forces = self.ctx.program(vertex_shader=vert_shader, fragment_shader=add_forces_shader)
        self.vbo = self.ctx.buffer(vertices.astype('f4').tobytes())
        self.vao = self.ctx.vertex_array(self.program, [(self.vbo, '2f 2f', 'vert', 'texcoord')])
        self.create_texture()
        self.sampler = self.ctx.sampler(False, False, False) 
        self.sampler.use()

    def render(self, time, frame_time):
        #return super().render(time, frame_time)
        #self.vao.render()
        
        frame_time = 1/60

        # Add Density
        self.ctx.clear()
        self.add_forces["field"] = 0
        self.add_forces["forces"] = 1
        self.field0.use(0)
        self.external_conditions.use(1)
        self.fboField1.use()
        self.quad.render(self.add_forces)
        self.swap_buffers()

        # Advect
        self.field0.use()
        self.fboField1.use()
        self.advect["dt"] = frame_time
        self.advect["imageSize"] = SCREEN_WIDTH
        self.quad.render(self.advect)
        self.swap_buffers()

        # Project
        #   - Divergence
        self.field0.use(0)
        self.fboPdiv0.use()
        self.quad.render(self.divergence)

        for i in range(20):
            self.pdiv0.use(0)
            self.fboPdiv1.use()
            self.quad.render(self.p_calc)
            self.pdiv0, self.pdiv1 = self.pdiv1, self.pdiv0
            self.fboPdiv0, self.fboPdiv1 = self.fboPdiv1, self.fboPdiv0

        self.last_prog["textura"] = 0
        self.last_prog["field"] = 1
        self.pdiv0.use(0)
        self.field0.use(1)
        self.fboField1.use()
        self.quad.render(self.last_prog)
        self.swap_buffers()

        # Spread
        # self.field0.use(0)
        # self.fboField1.use()
        # self.spread["dt"] = frame_time
        # self.spread["WIDTH"] = SCREEN_HEIGHT
        # self.quad.render(self.spread)

        # Render
        self.field1.use(0)
        self.ctx.screen.use()
        self.quad.render(self.program)

    def swap_buffers(self):
        self.field0, self.field1 = self.field1, self.field0
        self.fboField0, self.fboField1 = self.fboField1, self.fboField0


    def create_texture(self):
        tex = np.zeros([self.field0.size[0], self.field0.size[1], 4]).astype(np.float32)
        self.field0.write(tex.ravel().tobytes())
        self.field1.write(tex.ravel().tobytes())
        tex2 = np.zeros([self.field0.size[0], self.field0.size[1], 4]).astype(np.float32)
        tex2[:, :, :] = 0
        mid, width = (SCREEN_WIDTH//2-5), 10
        tex2[mid:mid+width, 25:30, 0] = 1.2/60
        tex2[mid:mid+width, 25:30, 2] = 10/60
        tex2[mid:mid+width, SCREEN_WIDTH-30:SCREEN_WIDTH-26, 0] = -1.4/60
        tex2[mid:mid+width, SCREEN_HEIGHT-30:SCREEN_WIDTH-26, 2] = 10/60
        self.external_conditions.write(tex2.ravel().tobytes())


mglw.run_window_config(TestWindow)
