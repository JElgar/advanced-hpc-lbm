#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

kernel void accelerate_flow(global t_speed* cells,
                            global int* obstacles,
                            int nx, int ny,
                            float density, float accel)
{
  /* compute weighting factors */
  float w1 = density * accel / 9.0;
  float w2 = density * accel / 36.0;

  /* modify the 2nd row of the grid */
  int jj = ny - 2;

  /* get column index */
  int ii = get_global_id(0);

  /* if the cell is not occupied and
  ** we don't send a negative density */
  if (!obstacles[ii + jj* nx]
      && (cells[ii + jj* nx].speeds[3] - w1) > 0.f
      && (cells[ii + jj* nx].speeds[6] - w2) > 0.f
      && (cells[ii + jj* nx].speeds[7] - w2) > 0.f)
  {
    /* increase 'east-side' densities */
    cells[ii + jj* nx].speeds[1] += w1;
    cells[ii + jj* nx].speeds[5] += w2;
    cells[ii + jj* nx].speeds[8] += w2;
    /* decrease 'west-side' densities */
    cells[ii + jj* nx].speeds[3] -= w1;
    cells[ii + jj* nx].speeds[6] -= w2;
    cells[ii + jj* nx].speeds[7] -= w2;
  }
}

kernel void prc(global t_speed* cells,
                      global t_speed* tmp_cells,
                      global int* obstacles,
                      int nx, int ny, 
                      float omega,
                      local float* l_tot_u_values,
                      global float* g_tot_u_values
                    )
{
  float tot_u = 0.f;

  /* get column and row indices */
  int ii = get_global_id(0);
  int jj = get_global_id(1);
  int local_id = get_local_id(0);
  int work_group_number = get_group_id(0);
  int local_size = get_local_size(0);

  /* determine indices of axis-direction neighbours
  ** respecting periodic boundary conditions (wrap around) */
  int y_n = (jj + 1) % ny;
  int x_e = (ii + 1) % nx;
  int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
  int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);
  
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float c_2sq = 2.f * c_sq; /* 2 times square of speed of sound */
  const float c_2cu = c_2sq * c_sq; /* 2 times cube of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  /* propagate densities from neighbouring cells, following
  ** appropriate directions of travel and writing into
  ** scratch space grid */

  if (obstacles[jj*nx + ii])
  {
    /* called after propagate, so taking values from scratch space
    ** mirroring, and writing into main grid */
    tmp_cells[ii + jj*nx].speeds[0] = cells[ii + jj*nx].speeds[0];
    tmp_cells[ii + jj*nx].speeds[1] = cells[x_e + jj*nx].speeds[3];
    tmp_cells[ii + jj*nx].speeds[2] = cells[ii + y_n*nx].speeds[4];
    tmp_cells[ii + jj*nx].speeds[3] = cells[x_w + jj*nx].speeds[1];
    tmp_cells[ii + jj*nx].speeds[4] = cells[ii + y_s*nx].speeds[2];
    tmp_cells[ii + jj*nx].speeds[5] = cells[x_e + y_n*nx].speeds[7];
    tmp_cells[ii + jj*nx].speeds[6] = cells[x_w + y_n*nx].speeds[8];
    tmp_cells[ii + jj*nx].speeds[7] = cells[x_w + y_s*nx].speeds[5];
    tmp_cells[ii + jj*nx].speeds[8] = cells[x_e + y_s*nx].speeds[6];
    l_tot_u_values[local_id] = 0;
  } 
  else
  {
    /* compute local density total */
    float local_density = 0.f;
    local_density += cells[ii + jj*nx].speeds[0];
    local_density += cells[x_w + jj*nx].speeds[1];
    local_density += cells[ii + y_s*nx].speeds[2];
    local_density += cells[x_e + jj*nx].speeds[3];
    local_density += cells[ii + y_n*nx].speeds[4];
    local_density += cells[x_w + y_s*nx].speeds[5];
    local_density += cells[x_e + y_s*nx].speeds[6];
    local_density += cells[x_e + y_n*nx].speeds[7];
    local_density += cells[x_w + y_n*nx].speeds[8];

    /* compute x velocity component */
    const float u_x = (
        cells[x_w + jj*nx].speeds[1]
      - cells[x_e + jj*nx].speeds[3]
      + cells[x_w + y_s*nx].speeds[5]
      - cells[x_e + y_s*nx].speeds[6]
      - cells[x_e + y_n*nx].speeds[7]
      + cells[x_w + y_n*nx].speeds[8]
    ) / local_density;

    /* compute y velocity component */
    const float u_y = (
        cells[ii + y_s*nx].speeds[2]
      - cells[ii + y_n*nx].speeds[4]
      + cells[x_w + y_s*nx].speeds[5]
      + cells[x_e + y_s*nx].speeds[6]
      - cells[x_e + y_n*nx].speeds[7]
      - cells[x_w + y_n*nx].speeds[8]
      ) / local_density;

    /* velocity squared */
    const float u_sq = u_x * u_x + u_y * u_y;
    const float u_sqd2sq = u_sq / c_2sq;

    /* relaxation step */
    tmp_cells[ii + jj*nx].speeds[0] = cells[ii + jj*nx].speeds[0]
                                           + omega
                                           * (
                                              w0 * local_density * (1.f - u_sq / (c_2sq))
                                              - cells[ii + jj*nx].speeds[0]
                                            );
    
    tmp_cells[ii + jj*nx].speeds[1] = cells[x_w + jj*nx].speeds[1]
                                           + omega
                                           * (
                                              w1 * local_density * (1.f + u_x / c_sq + (u_x * u_x) / c_2cu - u_sqd2sq)
                                              - cells[x_w + jj*nx].speeds[1]
                                            );
    
    tmp_cells[ii + jj*nx].speeds[2] = cells[ii + y_s*nx].speeds[2]
                                           + omega
                                           * (
                                              w1 * local_density * (1.f + u_y / c_sq + (u_y * u_y) / c_2cu - u_sqd2sq)
                                              - cells[ii + y_s*nx].speeds[2]
                                            );
    
    tmp_cells[ii + jj*nx].speeds[3] = cells[x_e + jj*nx].speeds[3]
                                           + omega
                                           * (
                                              w1 * local_density * (1.f - u_x / c_sq + (u_x * u_x) / c_2cu - u_sqd2sq) 
                                              - cells[x_e + jj*nx].speeds[3]
                                            );
    
    tmp_cells[ii + jj*nx].speeds[4] = cells[ii + y_n*nx].speeds[4]
                                           + omega
                                           * (
                                              w1 * local_density * (1.f - u_y / c_sq + (u_y * u_y) / c_2cu - u_sqd2sq) 
                                              - cells[ii + y_n*nx].speeds[4]
                                            );
    
    tmp_cells[ii + jj*nx].speeds[5] = cells[x_w + y_s*nx].speeds[5]
                                           + omega
                                           * (
                                              w2 * local_density * (1.f + (u_x + u_y) / c_sq + ((u_x + u_y) * (u_x + u_y)) / c_2cu - u_sqd2sq)
                                              - cells[x_w + y_s*nx].speeds[5]
                                            );
    
    tmp_cells[ii + jj*nx].speeds[6] = cells[x_e + y_s*nx].speeds[6]
                                           + omega
                                           * (
                                              w2 * local_density * (1.f + (-u_x + u_y) / c_sq + ((-u_x + u_y) * (-u_x + u_y))  / c_2cu - u_sqd2sq)
                                              - cells[x_e + y_s*nx].speeds[6]
                                            );
    
    tmp_cells[ii + jj*nx].speeds[7] = cells[x_e + y_n*nx].speeds[7]
                                           + omega
                                           * (
                                              w2 * local_density * (1.f - (u_x + u_y) / c_sq + ((u_x + u_y) * (u_x + u_y))    / c_2cu - u_sqd2sq) 
                                              - cells[x_e + y_n*nx].speeds[7]
                                            );
    
    tmp_cells[ii + jj*nx].speeds[8] = cells[x_w + y_n*nx].speeds[8]
                                           + omega
                                           * (
                                              w2 * local_density * (1.f + (u_x - u_y) / c_sq + ((u_x - u_y) * (u_x - u_y)) / c_2cu - u_sqd2sq)
                                              - cells[x_w + y_n*nx].speeds[8]
                                            );

    l_tot_u_values[local_id] = native_sqrt((u_x * u_x) + (u_y * u_y));
  }

  // printf("mylocal id is: %d, my work_group_number is: %d, local_size: %d\n", local_id, work_group_number, local_size);
  // all threads execute this code simultaneously:
  // Calculate wg tot_u value from list of local tot_u values 
  for( int offset = 1; offset < local_size; offset *= 2 )
  {
    int mask = 2*offset - 1;
    barrier( CLK_LOCAL_MEM_FENCE ); // wait for all threads to get here
    if( ( local_id & mask ) == 0 ) // bit-by-bit and’ing tells us which
    { // threads need to do work now
      l_tot_u_values[ local_id ] += l_tot_u_values[ local_id + offset ];
    }
  }
  
  barrier( CLK_LOCAL_MEM_FENCE );

  // Set wg tot_u value in global list
  if( local_id == 0 ) g_tot_u_values[ work_group_number ] = l_tot_u_values[ 0 ];
}
