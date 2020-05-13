__kernel void advect(
    __global float* field_x,    // lon, Deg E (-180 to 180)
    const unsigned int x_len,
    __global float* field_y,    // lat, Deg N (-90 to 90)
    const unsigned int y_len,
    __global float* field_U,    // m / s
    __global float* field_V,    // m / s
    __global float* x0,         // lon, Deg E (-180 to 180)
    __global float* y0,         // lat, Deg N (-90 to 90)
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
    for (unsigned int t_idx=0; t_idx<ntimesteps; t_idx++) {

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

        // find U and V nearest to particle position
        float u = field_U[x_idx*y_len + y_idx];
        float v = field_V[x_idx*y_len + y_idx];

        //////////// advect particle
        // meters displacement
        float dx_meters = u * dt;
        float dy_meters = v * dt;

        // convert meters displacement to lat/lon (Reference: American Practical Navigator, Vol II, 1975 Edition, p 5)
        float rlat = y * M_PI/180;
        float dx_deg = dx_meters / (111415.13 * cos(rlat) - 94.55 * cos(3 * rlat));
        float dy_deg = dy_meters / (111132.09 - 556.05 * cos(2 * rlat) + 1.2 * cos(4 * rlat));

        // update
        x = x + dx_deg;
        y = y + dy_deg;
        // keep latitude representation within [-180, 180)
        x = fmod(x + 180, 360) - 180;

        // save if necessary
        if ((t_idx+1) % save_every == 0) {
            unsigned int out_idx = (t_idx+1)/save_every - 1;
            X_out[p_id*out_timesteps + out_idx] = x;
            Y_out[p_id*out_timesteps + out_idx] = y;
        }
    }
}
