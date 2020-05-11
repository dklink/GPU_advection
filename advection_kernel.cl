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