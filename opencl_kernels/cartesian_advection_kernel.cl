__kernel void advect(
    __global float* field_x,
    const unsigned int x_len,
    __global float* field_y,
    const unsigned int y_len,
    __global float* field_t,    // time, same units as dt
    const unsigned int t_len,
    __global float* field_U,    // time unit same as dt
    __global float* field_V,    // ^^
    __global float* x0,         //
    __global float* y0,         //
    __global float* t0,         // same units as t0 and dt
    const float dt,
    const unsigned int ntimesteps,
    const unsigned int save_every,
    __global float* X_out,      // lon, Deg E (-180 to 180)
    __global float* Y_out)      // lat, Deg N (-90 to 90)
{
    int p_id = get_global_id(0);  // id of particle
    const unsigned int out_timesteps = ntimesteps / save_every;

    // loop timesteps
    float x = x0[p_id];
    float y = y0[p_id];
    float t = t0[p_id];
    for (unsigned int timestep=0; timestep<ntimesteps; timestep++) {

        // find index of nearest x
        unsigned int x_idx = 0;
        float min_distance = -1;
        for (unsigned int i=0; i<x_len; i++) {
            float distance = fabs((float)(field_x[i] - x));
            if ((distance < min_distance) || (min_distance == -1)) {
               min_distance = distance;
               x_idx = i;
            }
        }

        // find index of nearest y
        unsigned int y_idx = 0;
        min_distance = -1;
        for (unsigned int i=0; i<y_len; i++) {
            float distance = fabs((float)(field_y[i] - y));
            if ((distance < min_distance) || (min_distance == -1)) {
               min_distance = distance;
               y_idx = i;
            }
        }

        // find index of nearest t
        unsigned int t_idx = 0;
        min_distance = -1;
        for (unsigned int i=0; i<t_len; i++) {
            float distance = fabs((float)(field_t[i] - t));
            if ((distance < min_distance) || (min_distance == -1)) {
               min_distance = distance;
               t_idx = i;
            }
        }

        // find U and V nearest to particle position
        float u = field_U[(t_idx*x_len + x_idx)*y_len + y_idx];
        float v = field_V[(t_idx*x_len + x_idx)*y_len + y_idx];

        // advect particle
        x = x + u * dt;
        y = y + v * dt;
        t = t + dt;

        // save if necessary
        if ((timestep+1) % save_every == 0) {
            unsigned int out_idx = (timestep+1)/save_every - 1;
            X_out[p_id*out_timesteps + out_idx] = x;
            Y_out[p_id*out_timesteps + out_idx] = y;
        }
    }
}
