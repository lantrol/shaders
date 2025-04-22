import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
#import pygame as pg
import time
from math import ceil

# pg.init()

# screen = pg.display.set_mode([800, 800])

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

GRID_WIDTH = 800
GRID_HEIGHT = 800

DT = 1/60
dt0 = DT * GRID_WIDTH
h = 1 / GRID_WIDTH

# X represents horizontal and Y vertical
# Arrays need to transpose to correct visualization

@cuda.jit(fastmath=True)
def addForces(vel_x, vel_y, forces_x, forces_y):
    i, j = cuda.grid(2)
    i += 1
    j += 1
    if i <= 0 or i >= GRID_WIDTH+1 or j <= 0 or j > GRID_HEIGHT:
        return
    vel_x[i, j] += forces_x[i, j]*DT
    vel_y[i, j] += forces_y[i, j]*DT

@cuda.jit(fastmath=True)
def advect(vel_x, vel_y, new_vel_x, new_vel_y):
    #dt0 = DT * GRID_WIDTH
    i, j = cuda.grid(2)
    i += 1
    j += 1
    if i <= 0 or i >= GRID_WIDTH+1 or j <= 0 or j > GRID_HEIGHT:
        return

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

@cuda.jit(fastmath=True)
def project_divergence(vel_x, vel_y, div, p):
    #h = 1 / GRID_WIDTH
    i, j = cuda.grid(2)
    i += 1
    j += 1
    if i <= 0 or i >= GRID_WIDTH+1 or j <= 0 or j > GRID_HEIGHT:
        return

    div[i, j] = -0.5 * h * (vel_x[i+1, j] - vel_x[i-1, j] + vel_y[i, j+1] - vel_y[i, j-1])
    p[i, j] = 0

@cuda.jit(fastmath=True)
def project_preasure(div, p, new_p):
    #h = 1 / GRID_WIDTH
    i, j = cuda.grid(2)
    i += 1
    j += 1
    if i <= 0 or i >= GRID_WIDTH+1 or j <= 0 or j > GRID_HEIGHT:
        return

    new_p[i, j] = (div[i, j] + p[i-1, j] + p[i+1, j] + p[i, j-1] + p[i, j+1])/4

@cuda.jit(fastmath=True)
def project_velocity(vel_x, vel_y, p):
    #h = 1 / GRID_WIDTH
    i, j = cuda.grid(2)
    i += 1
    j += 1
    if i <= 0 or i >= GRID_WIDTH+1 or j <= 0 or j > GRID_HEIGHT:
        return
    
    vel_x[i, j] -= 0.5*(p[i+1, j]-p[i-1, j])/h
    vel_y[i, j] -= 0.5*(p[i, j+1]-p[i, j-1])/h


def main():
    vel_x = cuda.to_device(np.zeros((GRID_HEIGHT+2, GRID_WIDTH+2), dtype="float"))
    vel_y = cuda.to_device(np.zeros((GRID_HEIGHT+2, GRID_WIDTH+2), dtype="float"))
    new_vel_x = cuda.to_device(np.zeros((GRID_HEIGHT+2, GRID_WIDTH+2), dtype="float"))
    new_vel_y = cuda.to_device(np.zeros((GRID_HEIGHT+2, GRID_WIDTH+2), dtype="float"))
    div = cuda.to_device(np.zeros((GRID_HEIGHT+2, GRID_WIDTH+2), dtype="float"))
    p = cuda.to_device(np.zeros((GRID_HEIGHT+2, GRID_WIDTH+2), dtype="float"))
    new_p = cuda.to_device(np.zeros((GRID_HEIGHT+2, GRID_WIDTH+2), dtype="float"))

    forces_x = np.zeros((GRID_HEIGHT+2, GRID_WIDTH+2), dtype="float")
    forces_y = np.zeros((GRID_HEIGHT+2, GRID_WIDTH+2), dtype="float")
    
    # Uncomment to add a little of randomness
    # for i in range(1, GRID_WIDTH+1):
    #     for j in range(1, GRID_HEIGHT+1):
    #         forces_x[i, j] = (np.random.randint(100)-50)/50000
    
    forces_x[4:10, 400:410] = 0.5
    forces_x[GRID_WIDTH-10:GRID_WIDTH-4, 400:410] = -0.5
    forces_x = cuda.to_device(forces_x)
    forces_y = cuda.to_device(forces_y)


    # Setting grid and block size 
    threadsperblock = (16, 16)
    blockspergrid = (
        ceil(GRID_WIDTH / threadsperblock[0]),
        ceil(GRID_HEIGHT / threadsperblock[1])
    )

    print("Threads per block: {} \nBlocks per grid: {}".format(threadsperblock, blockspergrid))

    running = True
    iter = 1
    start = time.time()
    while running:
        addForces[blockspergrid, threadsperblock](vel_x, vel_y, forces_x, forces_y)
        #cuda.synchronize()
        advect[blockspergrid, threadsperblock](vel_x, vel_y, new_vel_x, new_vel_y)
        #cuda.synchronize()
        vel_x, new_vel_x = new_vel_x, vel_x
        vel_y, new_vel_y = new_vel_y, vel_y
        project_divergence[blockspergrid, threadsperblock](vel_x, vel_y, div, p)
        for k in range(20):
            project_preasure[blockspergrid, threadsperblock](div, p, new_p)
            p, new_p = new_p, p
        project_velocity[blockspergrid, threadsperblock](vel_x, vel_y, p)
        #cuda.synchronize()
        
        # for event in pg.event.get():
        #     if event.type == pg.QUIT:
        #         running = False

        if iter % 2000 == 0:
            end = time.time()
            print("Elapsed time, second run: {} || Frames per second: {}".format((end-start), 2000/(end-start)))
            plt.imshow(abs(np.transpose(vel_x.copy_to_host()*10)), interpolation='none', cmap='gray', vmin=0, vmax=0.5)
            plt.show(block=True)
            continuar = input("Continuar: ")
            if continuar == "n":
                return
            start = time.time()
        iter += 1
        
        # surf = pg.surfarray.make_surface(np.abs(vel_x)*255*5)
        # # Fill the background with white
        # screen.fill((255, 255, 255))
        # # Flip the display
        # screen.blit(surf, (0,0))
        # pg.display.flip()

    end = time.time()
    print("Elapsed time, second run: %s" % iter/(end-start))




main()
