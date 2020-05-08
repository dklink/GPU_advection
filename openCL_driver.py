import pyopencl as cl
import numpy as np
import time

# kernel for advecting particle
from Field2D import Field2D

kernelsource = """
__kernel void advect(
    __global float* field_x,
    const unsigned int x_len,
    __global float* field_y,
    const unsigned int y_len,
    __global float* field_U,
    __global float* field_V,
    __global float* x0,
    __global float* y0,
    const float dt,
    const unsigned int ntimesteps,
    __global float* X_out,
    __global float* Y_out)
{
    
    int p_id = get_global_id(0);  // id of particle
    
    // loop timesteps
    X_out[p_id*ntimesteps] = x0[p_id];
    Y_out[p_id*ntimesteps] = y0[p_id];

    for (int t_idx=0; t_idx<ntimesteps-1; t_idx++) {
    
        // find index of nearest x
        unsigned int x_idx = 0;
        float min_distance = -1;
        for (unsigned int i=0; i<x_len; i++) {
            float distance = fabs((float)(field_x[i] - X_out[p_id*ntimesteps + t_idx]));
            if ((distance < min_distance) || (min_distance == -1)) {
               min_distance = distance;
               x_idx = i;
            }
        }
        
        // find index of nearest y
        unsigned int y_idx = 0;
        min_distance = -1;
        for (unsigned int i=0; i<y_len; i++) {
            float distance = fabs((float)(field_y[i] - Y_out[p_id*ntimesteps + t_idx]));
            if ((distance < min_distance) || (min_distance == -1)) {
               min_distance = distance;
               y_idx = i;
            }
        }
        
        // find U and V nearest to particle position
        float u = field_U[x_idx*y_len + y_idx];
        float v = field_V[x_idx*y_len + y_idx];
        
        // advect particle
        X_out[p_id*ntimesteps + t_idx+1] = X_out[p_id*ntimesteps + t_idx] + u * dt;
        Y_out[p_id*ntimesteps + t_idx+1] = Y_out[p_id*ntimesteps + t_idx] + v * dt;
        
    }
    
}
"""


def openCL_advect(field: Field2D, p0, num_timesteps, dt, device_index=0):
    """
    :param field: object storing vector field/axes.  Only supports singleton time dimension for now.
    :param p0: initial positions of particles, numpy array shape (num_particles, 2)
    :param num_timesteps: how many timesteps are we advecting
    :param dt: width of timestep, same units as vectors in 'field'
    :param device_index: 0=cpu, 1=integrated GPU, 2=dedicated GPU.  this is on my hardware, not portable.
    :return: (P, buffer_seconds, kernel_seconds): (numpy array with advection paths, shape (num_particles, num_timesteps, 2),
                                                   time it took to transfer memory to/from device,
                                                   time it took to execute kernel on device)
    """
    num_particles = p0.shape[0]

    # Create a compute context
    # Ask the user to select a platform/device on the CLI
    context = cl.create_some_context(answers=['1', str(device_index)])

    # Create a command queue
    queue = cl.CommandQueue(context)

    # Create the compute program from the source buffer
    # and build it
    program = cl.Program(context, kernelsource).build()

    # initialize host vectors
    h_field_x = field.x.astype(np.float32)
    h_field_y = field.y.astype(np.float32)
    h_field_U = field.U.flatten().astype(np.float32)
    h_field_V = field.V.flatten().astype(np.float32)
    h_x0 = p0[:, 0].astype(np.float32)
    h_y0 = p0[:, 1].astype(np.float32)
    h_X_out = np.zeros(num_particles * num_timesteps).astype(np.float32)
    h_Y_out = np.zeros(num_particles * num_timesteps).astype(np.float32)

    buf_time = time.time()
    # Create the input arrays in device memory and copy data from host
    d_field_x = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_field_x)
    d_field_y = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_field_y)
    d_field_U = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_field_U)
    d_field_V = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_field_V)
    d_x0 = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_x0)
    d_y0 = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_y0)
    # Create the output arrays in device memory
    d_X_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, h_X_out.nbytes)
    d_Y_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, h_Y_out.nbytes)
    buf_time = time.time() - buf_time

    # Execute the kernel over the entire range of our 1d input
    # allowing OpenCL runtime to select the work group items for the device
    advect = program.advect
    advect.set_scalar_arg_dtypes(
            [None, np.uint32, None, np.uint32, None, None, None, None, np.float32, np.uint32, None, None])
    kernel_time = time.time()
    advect(queue, (num_particles,), None,
           d_field_x, np.uint32(len(h_field_x)),
           d_field_y, np.uint32(len(h_field_y)),
           d_field_U, d_field_V,
           d_x0, d_y0,
           np.float32(dt), np.uint32(num_timesteps),
           d_X_out, d_Y_out)

    # Wait for the commands to finish before reading back
    queue.finish()
    kernel_time = time.time() - kernel_time

    # Read back the results from the compute device
    tic = time.time()
    cl.enqueue_copy(queue, h_X_out, d_X_out)
    cl.enqueue_copy(queue, h_Y_out, d_Y_out)
    buf_time += time.time() - tic

    P = np.zeros([num_particles, num_timesteps, 2])
    P[:, :, 0] = h_X_out.reshape([num_particles, num_timesteps])
    P[:, :, 1] = h_Y_out.reshape([num_particles, num_timesteps])

    return P, buf_time, kernel_time
