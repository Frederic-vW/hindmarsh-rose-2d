#!/usr/local/bin/julia
# last tested Julia version: 1.6.1
# Hindmarsh-Rose model on a 2D lattice
# FvW 03/2018

using NPZ
using PyCall
using PyPlot
using Statistics
using VideoIO
@pyimport matplotlib.animation as anim

function animate_pyplot(fname, data)
    """
    Animate 3D array as .mp4 using PyPlot, save as `fname`
    array dimensions:
        1: time
        2, 3: space
    """
    nt, nx, ny = size(data)
    vmin = minimum(data)
    vmax = maximum(data)
    # setup animation image
    fig = figure(figsize=(6,6))
    axis("off")
    t = imshow(data[1,:,:], origin="lower", cmap=ColorMap("gray"),
			   vmin=vmin, vmax=vmax)
    tight_layout()
    # frame generator
    println("[+] animate")
    function animate(i)
        (i%100 == 0) && print("    t = ", i, "/", nt, "\r")
        t.set_data(data[i+1,:,:])
    end
    # create animation
    ani = anim.FuncAnimation(fig, animate, frames=nt, interval=10)
    println("\n")
    # save animation
    ani[:save](fname, bitrate=-1,
               extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    show()
end

function animate_video(fname, data)
    """
    Animate 3D array as .mp4 using VideoIO, save as `fname`
    array dimensions:
        1: time
        2, 3: space
    """
    # BW
    y = UInt8.(round.(255*(data .- minimum(data))/(maximum(data)-minimum(data))))
    # BW inverted
    #y = UInt8.(round.(255 .- 255*(data .- minimum(data))/(maximum(data)-minimum(data))))
    encoder_options = (color_range=2, crf=0, preset="medium")
    framerate=90
    T = size(data,1)
    open_video_out(fname, y[1,end:-1:1,:], framerate=framerate,
                   encoder_options=encoder_options) do writer
        for i in range(2,stop=T,step=1)
            write(writer, y[i,end:-1:1,:])
        end
    end
end

function L(x)
    # Laplace operator
    # periodic boundary conditions
    xU = circshift(x, [-1 0])
    xD = circshift(x, [1 0])
    xL = circshift(x, [0 -1])
    xR = circshift(x, [0 1])
    Lx = xU + xD + xL + xR - 4x
    # non-periodic boundary conditions
    Lx[1,:] .= 0.0
    Lx[end,:] .= 0.0
    Lx[:,1] .= 0.0
    Lx[:,end] .= 0.0
    return Lx
end

function hr2d(N, T, t0, dt, sd, D, a, b, c, d, s, r, x_1, I0, stim, blocks)
	# initialize Hindmarsh-Rose system
	x, y, z = zeros(Float64,N,N), zeros(Float64,N,N), zeros(Float64,N,N)
	dxdt, dydt, dzdt = zeros(Float64,N,N), zeros(Float64,N,N), zeros(Float64,N,N)
	sd_sqrt_dt = sd*sqrt(dt)
	X = zeros(Float64,T,N,N)
	I = zeros(Float64,t0+T,N,N)
	# stimulation protocol
	for st in stim
		t_on, t_off = st[1]
		x0, x1 = st[2]
		y0, y1 = st[3]
		I[t0+t_on:t0+t_off, x0:x1, y0:y1] .= I0
		I[1:t0, x0:x1, y0:y1] .= I0
	end
	# iterate
	for t in range(1, stop=t0+T, step=1)
		(t%100 == 0) && print("    t = ", t, "\r")
		# Hindmarsh-Rose equations
		dx = -a*x.*x.*x + b*x.*x + y - z + I[t,:,:] + D*L(x)
		dy = c .- d*x.*x - y
		dz = r*(s*(x .- x_1) - z)
		# Ito stochastic integration
		x += (dx*dt + sd_sqrt_dt*randn(N,N))
		y += (dy*dt)
		z += (dz*dt)
		# dead block(s):
		for bl in blocks
			x[bl[1][1]:bl[1][2], bl[2][1]:bl[2][2]] .= 0.0
			y[bl[1][1]:bl[1][2], bl[2][1]:bl[2][2]] .= 0.0
			z[bl[1][1]:bl[1][2], bl[2][1]:bl[2][2]] .= 0.0
		end
		(t > t0) && (X[t-t0,:,:] = x)
	end
	println("\n")
	return X
end

function L(x)
	# Laplace operator
    xU = circshift(x, [-1 0])
    xD = circshift(x, [1 0])
    xL = circshift(x, [0 -1])
    xR = circshift(x, [0 1])
    Lx = xU + xD + xL + xR - 4x
	nx = size(x)[1]
	ny = size(x)[2]
    Lx[1,:] .= 0.0
	Lx[nx,:] .= 0.0
	Lx[:,1] .= 0.0
	Lx[:,ny] .= 0.0
    return Lx
end

function main()
    println("[+] Hindmarsh-Rose 2D lattice:")
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
	r = 0.001
    x_1 = -1.6
	I0 = 3.50 # 1.3
	#I1 = 0.05 # 1.3
    println("[+] Lattice size N: ", N)
	println("[+] Time steps T: ", N)
    println("[+] Warm-up steps t0: ", t0)
	println("[+] Integration time step dt: ", dt)
	println("[+] Noise intensity sd: ", sd)
	println("[+] Diffusion constant D: ", D)
    println("[+] HR parameter a: ", a)
    println("[+] HR parameter b: ", b)
    println("[+] HR parameter c: ", c)
    println("[+] HR parameter d: ", d)
    println("[+] HR parameter s: ", s)
    println("[+] HR parameter r: ", r)
    println("[+] HR parameter x_1: ", x_1)
    println("[+] Stimulation current I0: ", I0)
    #println("[+] Stimulation current I1: ", I1)

	# stim protocol, array of elements [[t0,t1], [x0,x1], [y0,y1]]
	stim = [ [[1,15000], [5,N-5], [5,N-5]] ]
    #stim = []# dead blocks, array of elementy [[x0,x1], [y0,y1]]

	# dead blocks, array of elementy [[x0,x1], [y0,y1]]
	#blocks = [ [[2*n_4,3*n_4], [15,20]], [[2*n_4+10,3*n_4+10], [40,45]] ]
	blocks = []

	# run simulation
	data = hr2d(N, T, t0, dt, sd, D, a, b, c, d, s, r, x_1, I0, stim, blocks)
    println("[+] Data dimensions: ", size(data))

	# plot mean voltage
    m = mean(reshape(data, (T,N*N)), dims=2)
    plot(m, "-k"); show()

	# save data
	I_str = rpad(I0, 4, '0') # stim. current amplitude as 4-char string
	s_str = rpad(s, 4, '0') # noise as 4-char string
	sd_str = rpad(sd, 4, '0') # noisy intensity as 4-char string
    D_str = rpad(D, 4, '0') # diffusion coefficient as 4-char string
	fname1 = string("hr2d_I_", I_str, "_s_", s_str, "_sd_", sd_str,
					"_D_", D_str, ".npy")
    #npzwrite(data_filename, data)
	#println("[+] Data saved as: ", data_filename)

	# video
	fname2 = string("hr2d_I_", I_str, "_s_", s_str, "_sd_", sd_str,
					"_D_", D_str, ".mp4")
	#animate_pyplot(fname2, data) # slow
    animate_video(fname2, data) # fast
	println("[+] Data saved as: ", fname2)
end

main()
