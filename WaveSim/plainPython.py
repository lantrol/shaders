from numba import cuda
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import pygame as pg
import time as clock

# pg.init()
# display = pg.display.set_mode((512, 109))

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

C = 3.4723*10**2
CELL_SIZE = 3.83 # mm
# TIME_STEP <= CELL_SIZE / sqrt(2*c)

plt.ion()

class FDTD2D():
    C = 343
    AIR_DENSITY = 1.225
    BASE_PRESSURE = 0
    DAMPING_COEF = 0.999

    def __init__(
        self,
        simSize = [0.05, 0.05],
        domainDim = [512, 512],
        fr = 40000,
        c = 343,
        rho = 1.225,
        nPML = 8,
        damp = 1
        ):

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

        self.waveLen = None
        self.ds = None # Cell size
        self.dt = None # Timestep
        self.T = None # Period

        # Fields
        self.p = None # Preassure
        self.vx = None
        self.vy = None
        self.att = None # Attenuation
        self.solid = None # Solid walls, particles or emitters

        # Precalc fields
        self.nSolidDampAtt = None
        self.attDt = None

        # Precalc complex fields for each emitter
        self.pressureForEachEmitter = None

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

        self.initPml()

    def initPml(self):
        # Initialize matching layer
        xDim = self.domainDim[0]
        yDim = self.domainDim[1]

        for i in range(self.nPML):
            attValue = (0.5/self.dt) * (1 - (i/self.nPML))
            self.att[i:xDim-1-i, i] = attValue # Top layer
            self.att[i:xDim-1-i, yDim-1-i] = attValue # Bottom layer
            self.att[i, i:yDim-i] = attValue # Left layer
            self.att[xDim-1-i, i:yDim-i] = attValue # Right layer

    def resetFields(self):
        self.p = np.zeros(self.domainDim, dtype=np.float32)
        self.vx = np.zeros(self.domainDim, dtype=np.float32)
        self.vy = np.zeros(self.domainDim, dtype=np.float32)

    def iterate(self):
        # Precalc
        self.nSolidDampAtt = (1 - self.solid) * self.damp / (1 + self.att * self.dt)
        self.attDt = (1 + self.att * self.dt)
        dsRhoDt = self.dt / self.ds / self.rho
        dsCCRhoDt = self.c*self.c*self.rho*self.dt / self.ds

        # Velocity update
        pShiftPx = np.roll(self.p, -1, axis=0) # Shift to obtain gradient
        pShiftPy = np.roll(self.p, -1, axis=1)
        self.vx = (self.vx - (pShiftPx-self.p) * dsRhoDt) * self.nSolidDampAtt
        self.vy = (self.vy - (pShiftPy-self.p) * dsRhoDt) * self.nSolidDampAtt

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
            self.vx[XX, YY] = velocityX
            self.vy[XX, YY] = velocityY

        # Pressure update
        vShiftNx = np.roll(self.vx, 1, axis=0)
        vShiftNy = np.roll(self.vy, 1, axis=1)
        self.p = (self.p - ((self.vx - vShiftNx) + (self.vy - vShiftNy)) * dsCCRhoDt) / self.attDt

        # Pressure emitters update
        for emitter in self.presEmitters:
            XX = emitter[0]
            YY = emitter[1]
            amp = emitter[2]
            fr = emitter[3]
            phase = emitter[4]

            amplitudeOfEmission = amp * np.sin(2 * np.pi * fr * self.timeStamp - phase)
            #inds = sub2ind(self.p.shape, XX, YY)
            self.p[XX, YY] = amplitudeOfEmission

        self.timeStamp = self.timeStamp + self.dt


    def calcAmplitudeAndPhase(self, warmUpScreens, measurePeriods, showMovingField, showFinalField):
        warmUpSteps = self.screensToSteps(warmUpScreens)
        measuringSteps = self.periodsToSteps(measurePeriods)

        print(warmUpSteps)
        print(measuringSteps)

        totalSteps = int(warmUpSteps+measuringSteps)
        ampField = np.zeros(self.domainDim, dtype=np.float64)
        phaseField = np.zeros(self.domainDim, dtype=np.float64)

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
                tmp1 = self.p > ampField
                ampField[tmp1] = self.p[tmp1]
                phaseField[tmp1] = np.mod(self.timeStamp, self.T) * (2 * np.pi / self.T)

        if showFinalField:
            print(ampField.max())
            print(phaseField.max())
            print("Time to completion: ", clock.time()-start)
            plt.imshow(self.p, interpolation='bilinear', aspect='auto')
            plt.show(block=True)
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

def sub2ind(array_shape, rows, cols):
    ind = cols*array_shape[1] + rows
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind


fdtd = FDTD2D()

fdtd.fr = 25e3
H = 7.5e-3

fdtd.setWidth(4e-2, H + 1e-3, 512)
fdtd.initialize()

# Add reflector
reflectorW = 38e-3
fdtd.solidLine(-reflectorW/2, H/2, reflectorW/2, H/2, 1)

# Add emiter
emitterW = 20e-3
fdtd.addLineEmitterAngle(0, -H/2, emitterW, 0, 1, fdtd.fr, 0)

#exit()

amp, phase = fdtd.calcAmplitudeAndPhase(2, 2, False, True)
print(fdtd.nPML)
