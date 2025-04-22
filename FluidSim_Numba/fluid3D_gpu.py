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

GRID_WIDTH = 64
GRID_HEIGHT = 64
GRID_DEPTH = 64

DT = 1/60
dt0 = DT * GRID_WIDTH
h = 1 / GRID_WIDTH

# X represents horizontal and Y vertical
# Arrays need to transpose to correct visualization

@cuda.jit('void(float32[:, :, :], float32[:, :, :], float32[:, :, :], float32[:, :, :], float32[:, :, :], float32[:, :, :], float32[:, :, :], float32[:, :, :])', fastmath=True)
def addForces(vel_x, vel_y, vel_z, dens, forces_x, forces_y, forces_z, dens_source):
    i, j, k = cuda.grid(3)
    if i <= 0 or i >= GRID_WIDTH-1 or j <= 0 or j >= GRID_HEIGHT-1 or k <= 0 or k >= GRID_DEPTH-1:
        return
    vel_x[i, j, k] += forces_x[i, j, k]*DT
    vel_y[i, j, k] += forces_y[i, j, k]*DT
    vel_z[i, j, k] += forces_z[i, j, k]*DT
    dens[i, j, k] += dens_source[i, j, k]*DT

@cuda.jit('void(float32[:, :, :], float32[:, :, :], float32[:, :, :], float32[:, :, :], float32[:, :, :], float32[:, :, :], float32[:, :, :], float32[:, :, :])', fastmath=True)
def advect(vel_x, vel_y, vel_z, dens, new_vel_x, new_vel_y, new_vel_z, new_dens):
    #dt0 = DT * GRID_WIDTH
    i, j, k = cuda.grid(3)
    if i >= GRID_WIDTH or j >= GRID_HEIGHT or k >= GRID_DEPTH:
        return
    if i < 1:
        new_vel_x[i, j, k] = -vel_x[1, j, k]
        new_vel_y[i, j, k] = vel_y[1, j, k]
        new_vel_z[i, j, k] = vel_z[1, j, k]
        return
    elif i >= GRID_WIDTH-1:
        new_vel_x[i, j, k] = -vel_x[GRID_WIDTH-2, j, k]
        new_vel_y[i, j, k] = vel_y[GRID_WIDTH-2, j, k]
        new_vel_z[i, j, k] = vel_z[GRID_WIDTH-2, j, k]
        return
    elif j < 1:
        new_vel_x[i, j, k] = vel_x[i, 1, k]
        new_vel_y[i, j, k] = -vel_y[i, 1, k]
        new_vel_z[i, j, k] = vel_z[i, 1, k]
        return
    elif j >= GRID_HEIGHT-1:
        new_vel_x[i, j, k] = vel_x[i, GRID_HEIGHT-2, k]
        new_vel_y[i, j, k] = -vel_y[i, GRID_HEIGHT-2, k]
        new_vel_z[i, j, k] = vel_z[i, GRID_HEIGHT-2, k]
        return
    elif k < 1:
        new_vel_x[i, j, k] = vel_x[i, j, 1]
        new_vel_y[i, j, k] = vel_y[i, j, 1]
        new_vel_z[i, j, k] = -vel_z[i, j, 1]
        return
    elif k >= GRID_DEPTH-1:
        new_vel_x[i, j, k] = vel_x[i, j, GRID_DEPTH-2]
        new_vel_y[i, j, k] = vel_y[i, j, GRID_DEPTH-2]
        new_vel_z[i, j, k] = -vel_z[i, j, GRID_DEPTH-2]
        return

    x = i - dt0*vel_x[i, j, k]
    y = j - dt0*vel_y[i, j, k]
    z = k - dt0*vel_z[i, j, k]
    if x < 0.5:
        x = 0.5
    if x > GRID_WIDTH-1.5:
        x = GRID_WIDTH-1.5
    if y < 0.5:
        y = 0.5
    if y > GRID_HEIGHT-1.5:
        y = GRID_HEIGHT-1.5
    if z < 0.5:
        z = 0.5
    if z > GRID_DEPTH-1.5:
        z = GRID_DEPTH-1.5
    i0 = int(x)
    i1 = i0 + 1
    j0 = int(y)
    j1 = j0 + 1
    k0 = int(z)
    k1 = k0 + 1
    s1 = x - i0
    s0 = 1 - s1
    t1 = y - j0
    t0 = 1 - t1
    u1 = z - k0
    u0 = 1 - u1
    new_vel_x[i, j, k] =    (s0*(t0*(u0*vel_x[i0, j0, k0] + u1*vel_x[i0, j0, k1])+t1*(u0*vel_x[i0, j1, k0] + u1*vel_x[i0, j1, k1])) +
                            s1*(t0*(u0*vel_x[i1, j0, k0] + u1*vel_x[i1, j0, k1])+t1*(u0*vel_x[i1, j1, k0] + u1*vel_x[i1, j1, k1])))
    new_vel_y[i, j, k] =    (s0*(t0*(u0*vel_y[i0, j0, k0] + u1*vel_y[i0, j0, k1])+t1*(u0*vel_y[i0, j1, k0] + u1*vel_y[i0, j1, k1]))  +
                            s1*(t0*(u0*vel_y[i1, j0, k0] + u1*vel_y[i1, j0, k1])+t1*(u0*vel_y[i1, j1, k0] + u1*vel_y[i1, j1, k1])))
    new_vel_z[i, j, k] =    (s0*(t0*(u0*vel_z[i0, j0, k0] + u1*vel_z[i0, j0, k1])+t1*(u0*vel_z[i0, j1, k0] + u1*vel_z[i0, j1, k1]))  +
                            s1*(t0*(u0*vel_z[i1, j0, k0] + u1*vel_z[i1, j0, k1])+t1*(u0*vel_z[i1, j1, k0] + u1*vel_z[i1, j1, k1])))
    new_dens[i, j, k] =     (s0*(t0*(u0*dens[i0, j0, k0] + u1*dens[i0, j0, k1])+t1*(u0*dens[i0, j1, k0] + u1*dens[i0, j1, k1]))  +
                            s1*(t0*(u0*dens[i1, j0, k0] + u1*dens[i1, j0, k1])+t1*(u0*dens[i1, j1, k0] + u1*dens[i1, j1, k1])))


@cuda.jit('void(float32[:, :, :], float32[:, :, :], float32[:, :, :], float32[:, :, :], float32[:, :, :])', fastmath=True)
def project_divergence(vel_x, vel_y, vel_z, div, p):
    #h = 1 / GRID_WIDTH
    i, j, k = cuda.grid(3)
    if i <= 0 or i >= GRID_WIDTH-1 or j <= 0 or j >= GRID_HEIGHT-1 or k <= 0 or k >= GRID_DEPTH-1:
        return
    div[i, j, k] = -0.5 * h * (vel_x[i+1, j, k] - vel_x[i-1, j, k] + vel_y[i, j+1, k] - vel_y[i, j-1, k] + vel_z[i, j, k+1] - vel_z[i, j, k-1])
    p[i, j, k] = 0

@cuda.jit('void(float32[:, :, :], float32[:, :, :], float32[:, :, :])', fastmath=True)
def project_preasure(div, p, new_p):
    #h = 1 / GRID_WIDTH
    i, j, k = cuda.grid(3)
    if i <= 0 or i >= GRID_WIDTH-1 or j <= 0 or j >= GRID_HEIGHT-1 or k <= 0 or k >= GRID_DEPTH-1:
        return
    x0 = max(i-1, 1)
    x1 = min(i+1, GRID_WIDTH-2)
    y0 = max(j-1, 1)
    y1 = min(j+1, GRID_WIDTH-2)
    z0 = max(k-1, 1)
    z1 = min(k+1, GRID_WIDTH-2)
    new_p[i, j, k] = (div[i, j, k] + p[x0, j, k] + p[x1, j, k] + p[i, y0, k] + p[i, y1, k] + p[i, j, z0] + p[i, j, z1])/6

@cuda.jit('void(float32[:, :, :], float32[:, :, :], float32[:, :, :], float32[:, :, :])', fastmath=True)
def project_velocity(vel_x, vel_y, vel_z, p):
    #h = 1 / GRID_WIDTH
    i, j, k = cuda.grid(3)
    if i <= 0 or i >= GRID_WIDTH-1 or j <= 0 or j >= GRID_HEIGHT-1 or k <= 0 or k >= GRID_DEPTH-1:
        return
    x0 = max(i-1, 1)
    x1 = min(i+1, GRID_WIDTH-2)
    y0 = max(j-1, 1)
    y1 = min(j+1, GRID_WIDTH-2)
    z0 = max(k-1, 1)
    z1 = min(k+1, GRID_WIDTH-2)
    vel_x[i, j, k] -= 0.5*(p[x1, j, k]-p[x0, j, k])/h
    vel_y[i, j, k] -= 0.5*(p[i, y1, k]-p[i, y0, k])/h
    vel_z[i, j, k] -= 0.5*(p[i, j, z1]-p[i, j, z0])/h


def main():
    vel_x = cuda.to_device(np.zeros((GRID_HEIGHT, GRID_WIDTH, GRID_DEPTH), dtype="float32"))
    vel_y = cuda.to_device(np.zeros((GRID_HEIGHT, GRID_WIDTH, GRID_DEPTH), dtype="float32"))
    vel_z = cuda.to_device(np.zeros((GRID_HEIGHT, GRID_WIDTH, GRID_DEPTH), dtype="float32"))
    dens = cuda.to_device(np.zeros((GRID_HEIGHT, GRID_WIDTH, GRID_DEPTH), dtype="float32"))
    new_vel_x = cuda.to_device(np.zeros((GRID_HEIGHT, GRID_WIDTH, GRID_DEPTH), dtype="float32"))
    new_vel_y = cuda.to_device(np.zeros((GRID_HEIGHT, GRID_WIDTH, GRID_DEPTH), dtype="float32"))
    new_vel_z = cuda.to_device(np.zeros((GRID_HEIGHT, GRID_WIDTH, GRID_DEPTH), dtype="float32"))
    new_dens = cuda.to_device(np.zeros((GRID_HEIGHT, GRID_WIDTH, GRID_DEPTH), dtype="float32"))
    div = cuda.to_device(np.zeros((GRID_HEIGHT, GRID_WIDTH, GRID_DEPTH), dtype="float32"))
    p = cuda.to_device(np.zeros((GRID_HEIGHT, GRID_WIDTH, GRID_DEPTH), dtype="float32"))
    new_p = cuda.to_device(np.zeros((GRID_HEIGHT, GRID_WIDTH, GRID_DEPTH), dtype="float32"))

    forces_x = np.zeros((GRID_HEIGHT, GRID_WIDTH, GRID_DEPTH), dtype="float32")
    forces_y = np.zeros((GRID_HEIGHT, GRID_WIDTH, GRID_DEPTH), dtype="float32")
    forces_z = np.zeros((GRID_HEIGHT, GRID_WIDTH, GRID_DEPTH), dtype="float32")
    dens_source = np.zeros((GRID_HEIGHT, GRID_WIDTH, GRID_DEPTH), dtype="float32")

    # Uncomment to add a little of randomness
    # for i in range(1, GRID_WIDTH+1):
    #     for j in range(1, GRID_HEIGHT+1):
    #         forces_x[i, j] = (np.random.randint(100)-50)/50000
    
    forces_y[GRID_DEPTH//2-1:GRID_DEPTH//2, 9:10, GRID_DEPTH//2-1:GRID_DEPTH//2] = 0.5
    forces_y[GRID_DEPTH//2-1:GRID_DEPTH//2, GRID_DEPTH-10-1:GRID_DEPTH-9-1, GRID_DEPTH//2-1:GRID_DEPTH//2] = -0.5
    dens_source[GRID_DEPTH//2-1:GRID_DEPTH//2, 9:10, GRID_DEPTH//2-1:GRID_DEPTH//2] = 0.5
    dens_source[GRID_DEPTH//2-1:GRID_DEPTH//2, GRID_DEPTH-10-1:GRID_DEPTH-9-1, GRID_DEPTH//2-1:GRID_DEPTH//2] = 0.5

    forces_x = cuda.to_device(forces_x)
    forces_y = cuda.to_device(forces_y)
    forces_z = cuda.to_device(forces_z)
    dens_source = cuda.to_device(dens_source)


    # Setting grid and block size 
    threadsperblock = (8, 8, 8)
    blockspergrid = (
        ceil(GRID_WIDTH / threadsperblock[0]),
        ceil(GRID_HEIGHT / threadsperblock[1]),
        ceil(GRID_DEPTH / threadsperblock[2])
    )

    print("Threads per block: {} \nBlocks per grid: {}".format(threadsperblock, blockspergrid))

    running = True
    iters = 1
    start = time.time()
    while running:
        addForces[blockspergrid, threadsperblock](vel_x, vel_y, vel_z, dens, forces_x, forces_y, forces_z, dens_source)
        #cuda.synchronize()
        advect[blockspergrid, threadsperblock](vel_x, vel_y, vel_z, dens, new_vel_x, new_vel_y, new_vel_z, new_dens)
        #cuda.synchronize()
        vel_x, new_vel_x = new_vel_x, vel_x
        vel_y, new_vel_y = new_vel_y, vel_y
        vel_z, new_vel_z = new_vel_z, vel_z
        dens, new_dens = new_dens, dens
        project_divergence[blockspergrid, threadsperblock](vel_x, vel_y, vel_z, div, p)
        for k in range(20):
            project_preasure[blockspergrid, threadsperblock](div, p, new_p)
            p, new_p = new_p, p
        project_velocity[blockspergrid, threadsperblock](vel_x, vel_y, vel_z, p)
        #cuda.synchronize()
        
        # for event in pg.event.get():
        #     if event.type == pg.QUIT:
        #         running = False

        if iters % 3900 == 0:
            end = time.time()
            print("Elapsed time, second run: {} || Frames per second: {}".format((end-start), 3900/(end-start)))
            plt.imshow(abs(np.transpose(dens[:, :, GRID_DEPTH//2].copy_to_host())), interpolation='none', cmap='gray', vmin=0, vmax=0.1)
            plt.show(block=True)
            continuar = input("Continuar: ")
            if continuar == "n":
                running = False
            start = time.time()
        iters += 1
        
        # surf = pg.surfarray.make_surface(np.abs(vel_x)*255*5)
        # # Fill the background with white
        # screen.fill((255, 255, 255))
        # # Flip the display
        # screen.blit(surf, (0,0))
        # pg.display.flip()

    end = time.time()
    print("Elapsed time, second run: {} || Frames per second: {}".format((end-start), 3900/(end-start)))




main()
