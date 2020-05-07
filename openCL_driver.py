import pyopencl as cl
import numpy as np
import time
import matplotlib.pyplot as plt

# kernel for advecting particle
import generate_field
from plot_advection import plot_advection

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

# ------------------------------------------------------------------------------

# Main procedure

# Create a compute context
# Ask the user to select a platform/device on the CLI
context = cl.create_some_context()

# Create a command queue
queue = cl.CommandQueue(context)

# Create the compute program from the source buffer
# and build it
program = cl.Program(context, kernelsource).build()

nparticles = 1000000
ntimesteps = 100
dt = 1  # seconds
field = generate_field.converge_diverge()

# intialize host vectors
h_field_x = field.x.astype(np.float32)
h_field_y = field.y.astype(np.float32)
h_field_U = field.U[0].flatten().astype(np.float32)  # only first timestep
h_field_V = field.V[0].flatten().astype(np.float32)
h_x0 = np.random.randint(np.min(h_field_x), np.max(h_field_x), nparticles).astype(np.float32)
h_y0 = np.random.randint(np.min(h_field_y), np.max(h_field_y), nparticles).astype(np.float32)
h_X_out = np.zeros(nparticles*ntimesteps).astype(np.float32)
h_Y_out = np.zeros(nparticles*ntimesteps).astype(np.float32)

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

# Start the timer
rtime = time.time()

# Execute the kernel over the entire range of our 1d input
# allowing OpenCL runtime to select the work group items for the device
advect = program.advect
advect.set_scalar_arg_dtypes([None, np.uint32, None, np.uint32, None, None, None, None, np.float32, np.uint32, None, None])
advect(queue, (nparticles,), None,
       d_field_x, np.uint32(len(h_field_x)),
       d_field_y, np.uint32(len(h_field_y)),
       d_field_U, d_field_V,
       d_x0, d_y0,
       np.float32(dt), np.uint32(ntimesteps),
       d_X_out, d_Y_out)

# Wait for the commands to finish before reading back
queue.finish()
rtime = time.time() - rtime
print("The kernel ran in", rtime, "seconds")

# Read back the results from the compute device
cl.enqueue_copy(queue, h_X_out, d_X_out)
cl.enqueue_copy(queue, h_Y_out, d_Y_out)

# Test the results
X_out = h_X_out.reshape([nparticles, ntimesteps])
Y_out = h_Y_out.reshape([nparticles, ntimesteps])

P = np.zeros([nparticles, ntimesteps, 2])
P[:, :, 0] = X_out
P[:, :, 1] = Y_out

time = np.arange(0, ntimesteps, 1)
plot_advection(P[:500], time, field)
