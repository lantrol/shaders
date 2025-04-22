from os import write
import moderngl
import moderngl_window as mglw
from moderngl_window import geometry
from moderngl_window.geometry.attributes import AttributeNames
import glm
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import time as clock

# pg.init()
# display = pg.display.set_mode((512, 109))

# WARNING:
# This version was made while converting the code to OpenGL
# A lot of memory movement is made from CPU to GPU and the other way around
# to verify everything works as intended.
# This makes the code significantly slower. The "OnlyOpengl2D.py" file
# makes all calculations on GPU memory, beeing a lot faster.

# I kept code commented to see what parts is OpenGL replacing
# of the original python code.

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

class FDTD2D():
    gl_version = (4,3)
    window_size = (SCREEN_WIDTH, SCREEN_HEIGHT)
    aspect_ratio = SCREEN_WIDTH/SCREEN_HEIGHT
    vsync = True

    def __init__(
        self,
        simSize = [0.05, 0.05],
        domainDim = [512, 512],
        fr = 40000,
        c = 343,
        rho = 1.225,
        nPML = 8,
        damp = 1,
        ):
        self.ctx = moderngl.create_context(standalone=True)
        self.simSize = simSize
        self.domainDim = domainDim
        self.fr = fr
        self.c = c
        self.rho = rho
        self.nPML = nPML
        self.damp = damp
        self.timeStamp = 0

        self.velEmitters = []
        self.presEmitters = []

        self.waveLen = float
        self.ds = float # Cell size
        self.dt = float # Timestep
        self.T = float # Period

        # Fields
        self.p = np.ndarray # Preassure
        self.vx = np.ndarray
        self.vy = np.ndarray
        self.att = np.ndarray # Attenuation
        self.solid = np.ndarray # Solid walls, particles or emitters

        # Precalc fields
        self.nSolidDampAtt = np.ndarray
        self.attDt = np.ndarray

        self.pGL = np.ndarray
        self.pGL1 = np.ndarray
        self.vel0 = np.ndarray
        self.vel1 = np.ndarray
        self.attGL = np.ndarray
        self.solidGL = np.ndarray
        self.nSolidDampAttGL = np.ndarray
        self.attDtGL = np.ndarray

        # Precalc complex fields for each emitter
        self.pressureForEachEmitter = None

        # Shaders
        self.shaderNSolidDampAtt = self.ctx.compute_shader(open("nSolidDampAtt.glsl").read())
        self.shaderAttDt = self.ctx.compute_shader(open("attDt.glsl").read())
        self.shaderVelUpdate = self.ctx.compute_shader(open("shiftVelUpdate.glsl").read())
        self.shaderVelEmitter = self.ctx.compute_shader(open("velEmitter.glsl").read())
        self.shaderPresUpdate = self.ctx.compute_shader(open("presUpdate.glsl").read())
        self.shaderMeasuringStep = self.ctx.compute_shader(open("measuringStep.glsl").read())

    def initialize(self):
        self.timeStamp = 0

        # Precalc
        self.waveLen = self.c/self.fr
        self.ds = self.simSize[0]/self.domainDim[0]
        self.dt = self.ds / (np.sqrt(2)*self.c) / 2
        self.T = 1/self.fr

        # Information
        print("Ds is {} wavelengts".format(self.ds/self.waveLen))
        print("Dt is {}".format(self.dt))
        ds = self.ds/self.waveLen
        dt = self.dt

        # Array allocation
        self.p = np.zeros(self.domainDim, dtype=np.float32)
        self.vx = np.zeros(self.domainDim, dtype=np.float32)
        self.vy = np.zeros(self.domainDim, dtype=np.float32)
        self.att = np.zeros(self.domainDim, dtype=np.float32)
        self.solid = np.zeros(self.domainDim, dtype=np.float32)
        self.nSolidDampAtt = np.zeros(self.domainDim, dtype=np.float32)
        self.attDt = np.zeros(self.domainDim, dtype=np.float32)

        self.pGL = self.ctx.texture((self.domainDim[1], self.domainDim[0]), 1, dtype="f4")
        self.pGL1 = self.ctx.texture((self.domainDim[1], self.domainDim[0]), 1, dtype="f4")
        self.vel0 = self.ctx.texture((self.domainDim[1], self.domainDim[0]), 2, dtype="f4")
        self.vel1 = self.ctx.texture((self.domainDim[1], self.domainDim[0]), 2, dtype="f4")
        self.attGL = self.ctx.texture((self.domainDim[1], self.domainDim[0]), 1, dtype="f4")
        self.solidGL = self.ctx.texture((self.domainDim[1], self.domainDim[0]), 1, dtype="f4")
        self.nSolidDampAttGL = self.ctx.texture((self.domainDim[1], self.domainDim[0]), 1, dtype="f4")
        self.attDtGL = self.ctx.texture((self.domainDim[1], self.domainDim[0]), 1, dtype="f4")
        self.ampPhaseGL0 = self.ctx.texture((self.domainDim[1], self.domainDim[0]), 2, dtype="f4")
        self.ampPhaseGL1 = self.ctx.texture((self.domainDim[1], self.domainDim[0]), 2, dtype="f4")
        # self.attDt = self.ctx.texture((self.domainDim[0], self.domainDim[1]), 3, dtype="f4")

        self.initPml()

        # Shader uniforms
        self.shaderNSolidDampAtt["width"] = self.domainDim[1]
        self.shaderNSolidDampAtt["height"] = self.domainDim[0]
        self.shaderAttDt["width"] = self.domainDim[1]
        self.shaderAttDt["height"] = self.domainDim[0]
        self.shaderVelUpdate["width"] = self.domainDim[1]
        self.shaderVelUpdate["height"] = self.domainDim[0]
        self.shaderPresUpdate["width"] = self.domainDim[1]
        self.shaderPresUpdate["height"] = self.domainDim[0]
        self.shaderMeasuringStep["width"] = self.domainDim[1]
        self.shaderMeasuringStep["height"] = self.domainDim[0]

    def initPml(self):
        # Initialize matching layer
        xDim = self.domainDim[0]
        yDim = self.domainDim[1]
        att = np.zeros(self.domainDim, dtype= np.float32)

        for i in range(self.nPML):
            attValue = (0.5/self.dt) * (1 - (i/self.nPML))
            att[i:xDim-1-i, i] = attValue # Top layer
            att[i:xDim-1-i, yDim-1-i] = attValue # Bottom layer
            att[i, i:yDim-i] = attValue # Left layer
            att[xDim-1-i, i:yDim-i] = attValue # Right layer

        self.att = np.copy(att)
        self.attGL.write(att.ravel().tobytes())

    def resetFields(self):
        self.p = np.zeros(self.domainDim, dtype=np.float32)
        self.vx = np.zeros(self.domainDim, dtype=np.float32)
        self.vy = np.zeros(self.domainDim, dtype=np.float32)

    def iterate(self):
        # Precalc
        group_x = int(np.ceil(self.domainDim[1]/4))
        group_y = int(np.ceil(self.domainDim[0]/4))

        #self.nSolidDampAtt = (1 - self.solid) * self.damp / (1 + self.att * self.dt)
        self.solidGL.write(self.solid.ravel().tobytes())
        self.attGL.write(self.att.ravel().tobytes())
        self.shaderNSolidDampAtt["damp"] = self.damp
        self.shaderNSolidDampAtt["dt"] = self.dt
        self.solidGL.use(0)
        self.attGL.use(1)
        self.nSolidDampAttGL.bind_to_image(2, read=False, write= True)
        self.shaderNSolidDampAtt.run(group_x=group_x, group_y=group_y)
        self.nSolidDampAtt = np.frombuffer(self.nSolidDampAttGL.read(), dtype=np.float32).reshape(self.domainDim[0], self.domainDim[1])


        # self.attDt = (1 + self.att * self.dt)
        self.attGL.use(0)
        self.attDtGL.bind_to_image(0, read=False, write=True)
        self.shaderAttDt["dt"] = self.dt
        self.shaderAttDt.run(group_x=group_x, group_y=group_y)
        #self.attDt = np.frombuffer(self.attDtGL.read(), dtype=np.float32).reshape(self.domainDim[0], self.domainDim[1])

        dsRhoDt = self.dt / self.ds / self.rho
        dsCCRhoDt = self.c*self.c*self.rho*self.dt / self.ds

        # Velocity update
        #pShiftPx = np.roll(self.p, -1, axis=0)
        # -->
        # self.pGL.write(self.p.ravel().tobytes())
        # self.pGL.use(0)
        # self.pGL1.bind_to_image(0, read=False, write=True)
        # self.shaderShift.run(group_x=group_x, group_y=group_y)
        # pShiftPx = np.frombuffer(self.pGL1.read(), dtype=np.float32).reshape(self.domainDim[0], self.domainDim[1])

        #pShiftPy = np.roll(self.p, -1, axis=1)
        # -->
        # self.pGL.write(self.p.ravel().tobytes())
        # self.pGL.use(0)
        # self.pGL1.bind_to_image(0, read=False, write=True)
        # self.shaderShift2.run(group_x=group_x, group_y=group_y)
        # pShiftPy = np.frombuffer(self.pGL1.read(), dtype=np.float32).reshape(self.domainDim[0], self.domainDim[1])


        # Both pShiftPy and pShiftPx calculations are now made inside the next shader at the same time

        # self.vx = (self.vx - (pShiftPx-self.p) * dsRhoDt) * self.nSolidDampAtt
        # self.vy = (self.vy - (pShiftPy-self.p) * dsRhoDt) * self.nSolidDampAtt
        # -->
        vel = np.zeros((self.domainDim[0], self.domainDim[1], 2), dtype=np.float32)
        vel[:, :, 0] = self.vx.copy()
        vel[:, :, 1] = self.vy.copy()
        self.pGL.write(self.p.ravel().tobytes())
        self.nSolidDampAttGL.write(self.nSolidDampAtt.ravel().tobytes())
        self.vel0.write(vel.ravel().tobytes())
        self.pGL.use(0)
        self.nSolidDampAttGL.use(1)
        self.vel0.use(2)
        self.vel1.bind_to_image(0, read=False, write=True)
        self.shaderVelUpdate["dsRhoDt"] = dsRhoDt
        self.shaderVelUpdate.run(group_x = group_x, group_y = group_y)
        vel = np.frombuffer(self.vel1.read(), dtype=np.float32).reshape(self.domainDim[0], self.domainDim[1], 2)
        self.vx = vel[:, :, 0].copy()
        self.vy = vel[:, :, 1].copy()



        #print(self.vx[149:200, 49:60])

        # Velocity emitters update
        for emitter in self.velEmitters:
            XX = emitter[0]
            YY = emitter[1]
            amp = emitter[2]
            fr = emitter[3]
            phase = emitter[4]
            angle = emitter[5]

            amplitudeOfEmission = amp * np.sin(2 * np.pi * fr * self.timeStamp - phase)

            velocityX = amplitudeOfEmission * np.sin(angle)
            velocityY = amplitudeOfEmission * np.cos(angle)

            # inds = sub2ind(self.vx.shape, XX, YY)
            # self.vx[XX, YY] = velocityX
            # self.vy[XX, YY] = velocityY
            # -->
            xxData = self.ctx.buffer(np.array(XX, dtype=np.int32))
            yyData = self.ctx.buffer(np.array(YY, dtype=np.int32))
            self.shaderVelEmitter["velX"] = velocityX
            self.shaderVelEmitter["velY"] = velocityY
            vel = np.zeros((self.domainDim[0], self.domainDim[1], 2), dtype=np.float32)
            vel[:, :, 0] = self.vx.copy()
            vel[:, :, 1] = self.vy.copy()
            self.vel1.write(vel.tobytes())
            self.vel1.bind_to_image(0, read=False, write=True)
            xxData.bind_to_storage_buffer(0)
            yyData.bind_to_storage_buffer(1)
            self.shaderVelEmitter.run(group_x=np.array(XX, dtype=np.int32).size)
            vel = np.frombuffer(self.vel1.read(), dtype=np.float32).reshape(self.domainDim[0], self.domainDim[1], 2)
            self.vx = vel[:, :, 0].copy()
            self.vy = vel[:, :, 1].copy()

        # Pressure update
        # vShiftNx = np.roll(self.vx, 1, axis=0)
        # vShiftNy = np.roll(self.vy, 1, axis=1)
        # self.p = np.float32((self.p - ((self.vx - vShiftNx) + (self.vy - vShiftNy)) * dsCCRhoDt) / self.attDt)
        # -->
        self.pGL.write(self.p.ravel().tobytes())
        self.pGL.use(0)
        self.attDtGL.use(1)
        self.vel1.use(2)
        self.pGL1.bind_to_image(0, read=False, write=True)
        self.shaderPresUpdate["dsCCRhoDt"] = dsCCRhoDt
        self.shaderPresUpdate.run(group_x = group_x, group_y = group_y)
        self.p = np.frombuffer(self.pGL1.read(), dtype=np.float32).reshape(self.domainDim[0], self.domainDim[1])

        self.timeStamp = self.timeStamp + self.dt


    def calcAmplitudeAndPhase(self, warmUpScreens, measurePeriods, showMovingField, showFinalField):
        warmUpSteps = self.screensToSteps(warmUpScreens)
        measuringSteps = self.periodsToSteps(measurePeriods)

        print(warmUpSteps)
        print(measuringSteps)

        totalSteps = int(warmUpSteps+measuringSteps)
        ampField = np.zeros(self.domainDim, dtype=np.float32)
        phaseField = np.zeros(self.domainDim, dtype=np.float32)
        self.ampPhaseGL0.write(np.zeros((self.domainDim[1], self.domainDim[0], 2), dtype=np.float32).ravel().tobytes())
        self.ampPhaseGL1.write(np.zeros((self.domainDim[1], self.domainDim[0], 2), dtype=np.float32).ravel().tobytes())

        start = clock.time()

        for i in tqdm(range(totalSteps)):
            self.iterate()

            if showMovingField:
                im.set_array(self.p)
                plt.pause(0.0001)
                # surf = pg.surfarray.make_surface(np.repeat((127+self.p*127/715)[..., np.newaxis], 3, axis=-1))
                # display.blit(surf, (0, 0))
                # pg.display.update()

            if i == 2:
                start = clock.time()

            if i > warmUpSteps:
                # tmp1 = self.p > ampField
                # ampField[tmp1] = self.p[tmp1]
                # phaseField[tmp1] = np.mod(self.timeStamp, self.T) * (2 * np.pi / self.T)
                # -->
                group_x = int(np.ceil(self.domainDim[1]/4))
                group_y = int(np.ceil(self.domainDim[0]/4))
                self.pGL.write(self.p.ravel().tobytes())
                self.pGL.use(0)
                self.ampPhaseGL0.use(1)
                self.ampPhaseGL1.bind_to_image(0, read=False, write=True)
                self.shaderMeasuringStep["T"] = self.T
                self.shaderMeasuringStep["timeStamp"] = self.timeStamp
                self.shaderMeasuringStep.run(group_x = group_x, group_y = group_y)
                self.ampPhaseGL0, self.ampPhaseGL1 = self.ampPhaseGL1, self.ampPhaseGL0

        ampPhase = np.frombuffer(self.ampPhaseGL0.read(), dtype=np.float32).reshape(self.domainDim[0], self.domainDim[1], 2)
        ampField = ampPhase[:, :, 0].copy()
        phaseField = ampPhase[:, :, 1].copy()

        if showFinalField:
            print(ampField.max())
            print(phaseField.max())
            print("Time to completion: ", clock.time()-start)
            # plt.imshow(self.p, interpolation='bilinear', aspect='auto')
            # plt.show(block=True)
            plt.imshow(ampField, interpolation='bilinear', cmap='hot', aspect='auto')
            plt.show(block=True)
            plt.imshow(phaseField, interpolation='bilinear', cmap='hsv', aspect='auto')
            plt.show(block=True)

        return ampField, phaseField


    def screensToSteps(self, screens):
        return np.ceil(screens*np.max(self.domainDim)*self.ds/self.c/self.dt)

    def periodsToSteps(self, periods):
        return np.ceil(periods * self.T/self.dt)

    # Primitives
    def solidLine(self, ax, ay, bx, by, color):
        pax, pay = self.worldToGrid(ax, ay)
        pbx, pby = self.worldToGrid(bx, by)
        X, Y = self.bresenham(pax, pay, pbx, pby)
        # inds = sub2ind(self.solid.shape, X, Y)
        self.solid[X, Y] = color

    def addLineEmitterAngle(self, x, y, width, angle, amp, fr, phase):
        dx = np.cos(angle) * width/2
        dy = np.sin(angle) * width/2

        self.addLineEmitter( x-dx,y-dy,x+dx,y+dy,amp,fr,phase);

    def addLineEmitter(self, ax, ay, bx, by, amp, fr, phase):
        pax, pay = self.worldToGrid(ax, ay)
        pbx, pby = self.worldToGrid(bx, by)
        X, Y = self.bresenham(pax, pay, pbx, pby)
        self.solid[X, Y] = 1

        angle = np.atan2(by-ay, bx-ax)

        emitter = [X, Y, amp, fr, phase, angle]
        self.velEmitters += [emitter]

    # Coordinate Transformations
    def worldToGrid(self, x, y):
        px = np.int32(np.floor((x/self.ds) + self.domainDim[0] / 2))-1
        py = np.int32(np.floor((y/self.ds) + self.domainDim[1] / 2))-1
        return px, py

    def gridToWorld(self, x, y):
        px = np.int32((x - (self.domainDim[0] / 2))*self.ds)-1
        py = np.int32((y - (self.domainDim[1] / 2))*self.ds)-1
        return px, py

    # Primitive Helpers?
    def bresenham(self, x1, y1, x2, y2):
        x1 = np.int16(x1)
        y1 = np.int16(y1)
        x2 = np.int16(x2)
        y2 = np.int16(y2)

        xn = np.float64(x2 - x1)
        yn = np.float64(y2 - y1)

        if np.abs(xn) > np.abs(yn):
            xc = np.arange(x1, x2+np.sign(xn), np.sign(xn), dtype=np.int16)
            if yn == 0:
                yc = y1+np.zeros([1, int(np.abs(xn)+1)], dtype=np.int16)
            else:
                step = np.abs(yn/xn)*np.sign(yn)
                yc = np.arange(np.float64(y1), np.float64(y2)+step, step, dtype=np.int16)
        else:
            yc = np.arange(y1, y2+np.sign(yn), np.sign(yn), dtype=np.int16)
            if xn == 0:
                xc = x1+np.zeros([1, int(np.abs(yn)+1)], dtype=np.int16)
            else:
                step = np.abs(xn/yn)*np.sign(xn)
                xc = np.arange(np.float64(x1), np.float64(x2)+step, step, dtype=np.int16)
        return xc.ravel(), yc.ravel()

    def setWidth(self, w, h, cellsW):
        ds = w/cellsW
        cellsH = int(np.round(h/ds))
        h = cellsH * ds
        self.simSize = [w, h]
        self.domainDim = [cellsW, cellsH]

    def on_render(self, time, frame_time):
        self.ctx.clear()
        self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)

def sub2ind(array_shape, rows, cols):
    ind = cols*array_shape[1] + rows
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind

#mglw.run_window_config(FDTD2D)

fdtd = FDTD2D()

fdtd.fr = 25e3
H = 7.5e-3

fdtd.setWidth(4e-2, H + 1e-3, 512)
fdtd.initialize()

#exit()

# Add reflector
reflectorW = 38e-3
fdtd.solidLine(-reflectorW/2, H/2, reflectorW/2, H/2, 1)

# Add emiter
emitterW = 20e-3
fdtd.addLineEmitterAngle(0, -H/2, emitterW, 0, 1, fdtd.fr, 0)

#exit()

amp, phase = fdtd.calcAmplitudeAndPhase(2, 2, False, True)
print(fdtd.nPML)
