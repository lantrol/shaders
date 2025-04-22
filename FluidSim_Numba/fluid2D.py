import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
import pygame as pg
import time

pg.init()

screen = pg.display.set_mode([800, 800])

USE_NOPYTHON = True
USE_PARALLEL = True

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

GRID_WIDTH = 800
GRID_HEIGHT = 800

DT = 1/60

# X represents horizontal and Y vertical
# Array indexing is flipped to follow this

@jit(nopython=USE_NOPYTHON, parallel=USE_PARALLEL)
def addForces(vel_x, vel_y, forces_x, forces_y):
    vel_x[:, :] += forces_x[:, :]*DT
    vel_y[:, :] += forces_y[:, :]*DT

@jit(nopython=USE_NOPYTHON, parallel=USE_PARALLEL)
def advect(vel_x, vel_y, new_vel_x, new_vel_y):
    dt0 = DT * GRID_WIDTH
    for i in prange(1, GRID_WIDTH+1):
        for j in prange(1, GRID_HEIGHT+1):
            x = i - dt0*vel_x[i, j]
            y = j - dt0*vel_y[i, j]
            if x < 0.5:
                x = 0.5
            if x > GRID_WIDTH+0.5:
                x = GRID_WIDTH+0.5
            if y < 0.5:
                y = 0.5
            if y > GRID_HEIGHT+0.5:
                y = GRID_HEIGHT+0.5
            i0 = int(x)
            i1 = i0 + 1
            j0 = int(y)
            j1 = j0 + 1
            s1 = x - i0
            s0 = 1 - s1
            t1 = y - j0
            t0 = 1 - t1
            new_vel_x[i, j] = s0*(t0*vel_x[i0, j0]+t1*vel_x[i0, j1]) + s1*(t0*vel_x[i1, j0]+t1*vel_x[i1, j1])
            new_vel_y[i, j] = s0*(t0*vel_y[i0, j0]+t1*vel_y[i0, j1]) + s1*(t0*vel_y[i1, j0]+t1*vel_y[i1, j1])

@jit(nopython=USE_NOPYTHON, parallel=USE_PARALLEL)
def project(vel_x, vel_y, div, p):
    h = 1 / GRID_WIDTH
    for i in prange(1, GRID_WIDTH+1):
        for j in prange(1, GRID_HEIGHT+1):
            div[i, j] = -0.5 * h * (vel_x[i+1, j] - vel_x[i-1, j] + vel_y[i, j+1] - vel_y[i, j-1])
            p[i, j] = 0

    # set_bnd

    for k in prange(20):
        for i in prange(1, GRID_WIDTH+1):
            for j in prange(1, GRID_HEIGHT+1):
                p[i, j] = (div[i, j] + p[i-1, j] + p[i+1, j] + p[i, j-1] + p[i, j+1])/4
        #set_bnd

    for i in prange(1, GRID_WIDTH+1):
        for j in prange(1, GRID_HEIGHT+1):
            vel_x[i, j] -= 0.5*(p[i+1, j]-p[i-1, j])/h
            vel_y[i, j] -= 0.5*(p[i, j+1]-p[i, j-1])/h


def main():
    vel_x = np.zeros((GRID_HEIGHT+2, GRID_WIDTH+2, 3), dtype="float")
    vel_y = np.zeros((GRID_HEIGHT+2, GRID_WIDTH+2), dtype="float")
    div = np.zeros((GRID_HEIGHT+2, GRID_WIDTH+2), dtype="float")
    p = np.zeros((GRID_HEIGHT+2, GRID_WIDTH+2), dtype="float")
    forces_x = np.zeros((GRID_HEIGHT+2, GRID_WIDTH+2), dtype="float")
    forces_y = np.zeros((GRID_HEIGHT+2, GRID_WIDTH+2), dtype="float")

    new_vel_x = np.zeros((GRID_HEIGHT+2, GRID_WIDTH+2, 3), dtype="float")
    new_vel_y = np.zeros((GRID_HEIGHT+2, GRID_WIDTH+2), dtype="float")


    for i in range(1, GRID_WIDTH+1):
        for j in range(1, GRID_HEIGHT+1):
            forces_x[i, j] = (np.random.randint(100)-50)/50000
    forces_x[4:10, 400:410] = 0.5
    forces_x[790:800, 400:410] = -0.5

    running = True
    iter = 0
    start = time.time()
    while running:
        addForces(vel_x[:,:,0], vel_y, forces_x, forces_y)
        advect(vel_x[:,:,0], vel_y, new_vel_x[:,:,0], new_vel_y)
        vel_x, new_vel_x = new_vel_x, vel_x
        vel_y, new_vel_y = new_vel_y, vel_y
        project(vel_x[:,:,0], vel_y, div, p)
        
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        if iter % 300 == 0 and False:
            end = time.time()
            print("Elapsed time, second run: %s" % (end-start))
            plt.imshow(abs(np.transpose(vel_x)), interpolation='none', cmap='gray', vmin=0, vmax=0.5)
            plt.show(block=True)
            start = time.time()
        iter += 1
        
        surf = pg.surfarray.make_surface(np.abs(vel_x)*255*5)
        # Fill the background with white
        screen.fill((255, 255, 255))
        # Flip the display
        screen.blit(surf, (0,0))
        pg.display.flip()

    end = time.time()
    print("Elapsed time, second run: %s" % iter/(end-start))




main()
