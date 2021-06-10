#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Hindmarsh-Rose model on a 2D lattice
# FvW 03/2018

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import interp1d
import cv2


def hr2d(N, T, t0, dt, sd, D, a, b, c, d, s, r, x_1, I0, stim, blocks):
    # initialize Hindmarsh-Rose system
    x = np.zeros((N,N))
    y = np.zeros((N,N))
    z = np.zeros((N,N))
    dxdt = np.zeros((N,N)) # -1.2*np.ones((N,N)) # np.zeros((N,N))
    dydt = np.zeros((N,N))
    dzdt = np.zeros((N,N))
    sd_sqrt_dt = sd*np.sqrt(dt)
    X = np.zeros((T,N,N))
    #X[0,:,:] = x  # initialize
    I = np.zeros((t0+T,N,N))
    # stimulation protocol
    for st in stim:
        t_on, t_off = st[0]
        x0, x1 = st[1]
        y0, y1 = st[2]
        I[t0+t_on:t0+t_off, x0:x1, y0:y1] = I0
        I[:t0, x0:x1, y0:y1] = I0
    # iterate
    for t in range(t0+T):
        if (t%100 == 0): print("    t = ", t, "\r", end="")
        # Hindmarsh-Rose equations
        dx = -a*x*x*x + b*x*x + y - z + I[t,:,:] + D*L(x)
        dy = c - d*x*x - y
        dz = r*(s*(x - x_1) - z)
        # Ito stochastic integration
        x += (dx*dt + sd_sqrt_dt*np.random.randn(N,N))
        y += (dy*dt)
        z += (dz*dt)
        # dead block(s):
        for bl in blocks:
            x[bl[1][1]:bl[1][2], bl[2][1]:bl[2][2]] = 0.0
            y[bl[1][1]:bl[1][2], bl[2][1]:bl[2][2]] = 0.0
            z[bl[1][1]:bl[1][2], bl[2][1]:bl[2][2]] = 0.0
        if (t >= t0):
            X[t-t0,:,:] = x
    print("\n")
    return X


def animate_pyplot1(fname, data, downsample=10):
    """
    Animate 3D array as .mp4 using matplotlib.animation.FuncAnimation
    modified from:
    https://matplotlib.org/stable/gallery/animation/simple_anim.html
    (Faster than animate_pyplot2)

    Args:
        fname : string
            save result in mp4 format as `fname` in the current directory
        data : array (nt,nx,ny) (time, space, space)
        downsample : integer, optional
            temporal downsampling factor

    """
    nt, nx, ny = data.shape
    n1 = nt
    if downsample:
        n1 = int(nt/downsample) # number of samples after downsampling
        t0 = np.arange(nt)
        f_ip = interp1d(t0, data, axis=0, kind='linear')
        t1 = np.linspace(0, nt-1, n1)
        data = f_ip(t1)
    vmin, vmax = data.min(), data.max()
    # setup animation image
    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()
    ax.axis("off")
    t = plt.imshow(data[0,:,:], origin="lower", cmap=plt.cm.gray, \
                   vmin=vmin, vmax=vmax)
    plt.tight_layout()
    # frame generator
    print("[+] animate")
    def animate(i):
        if (i%10 == 0): print(f"    i = {i:d}/{n1:d}\r", end="")
        t.set_data(data[i,:,:])
    # create animation
    ani = animation.FuncAnimation(fig, animate, frames=n1, interval=10)
    #ani.save(fname)
    writer = animation.FFMpegWriter(fps=15, bitrate=1200)
    ani.save(fname, writer=writer)
    plt.show()
    print("\n")


def animate_pyplot2(fname, data, downsample=10):
    """
    Animate 3D array as .mp4 using matplotlib.animation.ArtistAnimation
    modified from:
    https://matplotlib.org/stable/gallery/animation/dynamic_image.html
    (Slower than animate_pyplot1)

    Args:
        fname : string
            save result in mp4 format as `fname` in the current directory
        data : array (nt,nx,ny) (time, space, space)
        downsample : integer, optional
            temporal downsampling factor

    """
    nt, nx, ny = data.shape
    n1 = nt
    if downsample:
        n1 = int(nt/downsample) # number of samples after downsampling
        t0 = np.arange(nt)
        f_ip = interp1d(t0, data, axis=0, kind='linear')
        t1 = np.linspace(0, nt-1, n1)
        data = f_ip(t1)
    print("[+] animate")
    vmin, vmax = data.min(), data.max()
    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()
    ax.axis("off")
    plt.tight_layout()
    ims = []
    for i in range(n1):
        if (i%10 == 0): print(f"    i = {i:d}/{n1:d}\r", end="")
        im = ax.imshow(data[i,:,:], origin="lower", cmap=plt.cm.gray, \
                       vmin=vmin, vmax=vmax, animated=True)
        if i == 0:
            ax.imshow(data[i,:,:], origin="lower", cmap=plt.cm.gray, \
                           vmin=vmin, vmax=vmax)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
    #ani.save(fname2)
    writer = animation.FFMpegWriter(fps=15, bitrate=1200)
    ani.save(fname, writer=writer)
    plt.show()
    print("\n")


def animate_video(fname, x, downsample=None):
    nt, nx, ny = x.shape
    n1 = nt
    if downsample:
        n1 = int(nt/downsample) # number of samples after downsampling
        t0 = np.arange(nt)
        f_ip = interp1d(t0, x, axis=0, kind='linear')
        t1 = np.linspace(0, nt-1, n1)
        x = f_ip(t1)
    # BW
    y = 255 * (x-x.min()) / (x.max()-x.min())
    # BW inverted
    #y = 255 * ( 1 - (x-x.min()) / (x.max()-x.min()) )
    y = y.astype(np.uint8)
    print(f"n1 = {n1:d}, nx = {nx:d}, ny = {ny:d}")
    # write video using opencv
    frate = 20
    out = cv2.VideoWriter(fname, \
                          cv2.VideoWriter_fourcc(*'mp4v'), \
                          frate, (nx,ny))
    for i in range(n1):
        print(f"i = {i:d}/{n1:d}\r", end="")
        img = np.ones((nx, ny, 3), dtype=np.uint8)
        for j in range(3): img[:,:,j] = y[i,::-1,:]
        out.write(img)
    out.release()
    print("")


def L(x):
    # Laplace operator
    # periodic boundary conditions
    xU = np.roll(x, shift=-1, axis=0)
    xD = np.roll(x, shift=1, axis=0)
    xL = np.roll(x, shift=-1, axis=1)
    xR = np.roll(x, shift=1, axis=1)
    Lx = xU + xD + xL + xR - 4*x
    # non-periodic boundary conditions
    Lx[0,:] = 0.0
    Lx[-1,:] = 0.0
    Lx[:,0] = 0.0
    Lx[:,-1] = 0.0
    return Lx


def main():
    print("[+] Hindmarsh-Rose 2D lattice:")
    N = 64
    T = 15000
    t0 = 2500
    dt = 0.05
    sd = 0.05
    D = 2.0
    a = 1.0
    b = 3.0
    c = 1.0
    d = 5.0
    s = 2.0 # s=1: spiking, s=4: strong accomodation
    r = 0.001 # 0.001
    x_1 = -1.6
    I0 = 3.5 # 1.3
    #I1 = 0.05 # 1.3
    print("[+] Lattice size N: ", N)
    print("[+] Time steps T: ", N)
    print("[+] Warm-up steps t0: ", t0)
    print("[+] Integration time step dt: ", dt)
    print("[+] Noise intensity sd: ", sd)
    print("[+] Diffusion constant D: ", D)
    print("[+] HR parameter a: ", a)
    print("[+] HR parameter b: ", b)
    print("[+] HR parameter c: ", c)
    print("[+] HR parameter d: ", d)
    print("[+] HR parameter s: ", s)
    print("[+] HR parameter r: ", r)
    print("[+] HR parameter x_1: ", x_1)
    print("[+] Stimulation current I0: ", I0)
    #print("[+] Stimulation current I1: ", I1)

    # auxiliary variables
    #n_2 = int(N/2) # 1/2 lattice size
    #n_4 = int(N/4) # 1/4 lattice size
    #n_5 = int(N/5) # 1/5 lattice size

    # stim protocol, array of elements [[t0,t1], [x0,x1], [y0,y1]]
    #stim = [ [[25,50], [1,N], [3,8]], [[130,150], [n_2-2,n_2+2], [10,25]] ]
    #stim = [ [[100,1000], [0,N], [3,8]] ]
    #stim = [ [[10,750], [0,5], [0,5]],
    #         [[1500,2500], [0,30], [0,15]] ]
    # traveling wave
    #stim = [ [[0,800], [0,5], [0,5]],
    #         [[2500,3500], [25,50], [0,15]] ]
    # traveling wave
    #stim = [ [[1200,2500], [25,50], [0,15]] ]
    # spiral waves 1st pulse
    #stim = [ [[1200,2500], [25,50], [0,15]],
    #         [[3500,4500], [30,40], [0,25]]]
    # ...
    #stim = [ [[1200,2500], [25,50], [0,15]],
    #         [[7500,8500], [25,30], [0,50]]]
    # spiral waves during 1st pulse
    #stim = [ [[100,2000], [25,50], [0,15]],
    #         [[4500,5500], [20,45], [0,25]],
    #         [[7000,9000], [20,45], [0,25]] ]
    stim = [ [[0,15000], [5,N-5], [5,N-5]] ]
    #stim = []

    # dead blocks, array of elementy [[x0,x1], [y0,y1]]
    #blocks = [ [[2*n_4,3*n_4], [15,20]], [[2*n_4+10,3*n_4+10], [40,45]] ]
    #blocks = [ [[0,10], [5,10]] ]
    blocks = []

    # run simulation
    data = hr2d(N, T, t0, dt, sd, D, a, b, c, d, s, r, x_1, I0, stim, blocks)
    print("[+] Data dimensions: ", data.shape)

    # plot mean voltage
    m = np.mean(np.reshape(data, (T,N*N)), axis=1)
    plt.figure(figsize=(12,4))
    plt.plot(m, "-k")
    plt.tight_layout()
    plt.show()

    # save data
    fname1 = f"hr2d_I_{I0:.2f}_s_{s:.2f}_sd_{sd:.2f}_D_{D:.2f}.npy"
    #np.save(fname1, data)
    #print("[+] Data saved as: ", fname1)

    # video
    fname2 = f"hr2d_I_{I0:.2f}_s_{s:.2f}_sd_{sd:.2f}_D_{D:.2f}.mp4"
    #animate_pyplot1(fname2, data, downsample=10)
    #animate_pyplot2(fname2, data, downsample=10)
    animate_video(fname2, data, downsample=10) # fastest
    print("[+] Data saved as: ", fname2)


if __name__ == "__main__":
    os.system("clear")
    main()
