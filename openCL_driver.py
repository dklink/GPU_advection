import pyopencl as cl
import numpy as np
import time

from Field2D import Field2D


def openCL_advect(field: Field2D, p0, num_timesteps, save_every, dt, device_index, verbose=False,
                  kernel='cartesian'):
    """
    :param field: object storing vector field/axes.
    :param p0: initial positions of particles, numpy array shape (num_particles, 2)
    :param num_timesteps: how many timesteps are we advecting
    :param save_every: how many timesteps between saving state.  Must divide num_timesteps.
    :param dt: width of timestep, same units as vectors in 'field'
    :param device_index: 0=cpu, 1=integrated GPU, 2=dedicated GPU.  this is on my hardware, not portable.
    :param verbose: determines whether to print buffer sizes and timing results
    :param kernel: select a kernel for advection. Current options: ['cartesian', 'lat_lon']
    :return: (P, buffer_seconds, kernel_seconds): (numpy array with advection paths, shape (num_particles, num_timesteps, 2),
                                                   time it took to transfer memory to/from device,
                                                   time it took to execute kernel on device)
    """
    num_particles = p0.shape[0]
    assert num_timesteps % save_every == 0, "save_every must divide num_timesteps"
    out_timesteps = num_timesteps//save_every
    t0 = 0  # start time
    # Create a compute context
    # Ask the user to select a platform/device on the CLI
    context = cl.create_some_context(answers=['1', str(device_index)])

    # Create a command queue
    queue = cl.CommandQueue(context)

    # Create the compute program from the source buffer
    # and build it
    if kernel == 'cartesian':
        program = cl.Program(context, open('opencl_kernels/cartesian_advection_kernel.cl').read()).build()
    elif kernel == 'lat_lon':
        program = cl.Program(context, open('opencl_kernels/lat_lon_advection_kernel.cl').read()).build()
    else:
        raise ValueError("Input a valid kernel")

    # initialize host vectors
    h_field_x = field.x.astype(np.float32)
    h_field_y = field.y.astype(np.float32)
    h_field_t = field.time.astype(np.float32)
    h_field_U = field.U.flatten().astype(np.float32)
    h_field_V = field.V.flatten().astype(np.float32)
    h_x0 = p0[:, 0].astype(np.float32)
    h_y0 = p0[:, 1].astype(np.float32)
    h_t0 = (t0 * np.ones(num_particles)).astype(np.float32)
    h_X_out = np.zeros(num_particles * out_timesteps).astype(np.float32)
    h_Y_out = np.zeros(num_particles * out_timesteps).astype(np.float32)

    if verbose:
        # print size of buffers
        for buf_name, buf_value in {'h_field_x': h_field_x, 'h_field_y': h_field_y, 'h_field_t': h_field_t,
                                    'h_field_U': h_field_U, 'h_field_V': h_field_V,
                                    'h_x0': h_x0, 'h_y0': h_y0, 'h_t0': h_t0,
                                    'h_X_out': h_X_out, 'h_Y_out': h_Y_out}.items():
            print(f'{buf_name}: {buf_value.nbytes / 1e6} MB')

    buf_time = time.time()
    # Create the input arrays in device memory and copy data from host
    d_field_x = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_field_x)
    d_field_y = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_field_y)
    d_field_t = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_field_t)
    d_field_U = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_field_U)
    d_field_V = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_field_V)
    d_x0 = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_x0)
    d_y0 = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_y0)
    d_t0 = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_t0)
    # Create the output arrays in device memory
    d_X_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, h_X_out.nbytes)
    d_Y_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, h_Y_out.nbytes)
    buf_time = time.time() - buf_time

    # Execute the kernel over the entire range of our 1d input
    # allowing OpenCL runtime to select the work group items for the device
    advect = program.advect
    advect.set_scalar_arg_dtypes([None, np.uint32, None, np.uint32, None, np.uint32,
                                  None, None,
                                  None, None, None,
                                  np.float32, np.uint32, np.uint32,
                                  None, None])
    kernel_time = time.time()
    advect(queue, (num_particles,), None,
           d_field_x, np.uint32(len(h_field_x)),
           d_field_y, np.uint32(len(h_field_y)),
           d_field_t, np.uint32(len(h_field_t)),
           d_field_U, d_field_V,
           d_x0, d_y0, d_t0,
           np.float32(dt), np.uint32(num_timesteps), np.uint32(save_every),
           d_X_out, d_Y_out)

    # Wait for the commands to finish before reading back
    queue.finish()
    kernel_time = time.time() - kernel_time

    # Read back the results from the compute device
    tic = time.time()
    cl.enqueue_copy(queue, h_X_out, d_X_out)
    cl.enqueue_copy(queue, h_Y_out, d_Y_out)
    buf_time += time.time() - tic

    # reshape results and store in numpy array
    P = np.zeros([num_particles, out_timesteps, 2])
    P[:, :, 0] = h_X_out.reshape([num_particles, out_timesteps])
    P[:, :, 1] = h_Y_out.reshape([num_particles, out_timesteps])

    if verbose:
        print(f'memory operations took {buf_time: .3f} seconds')
        print(f'kernel execution took  {kernel_time: .3f} seconds')

    return P, buf_time, kernel_time
