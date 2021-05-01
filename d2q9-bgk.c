/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

/**
 *
 * TODO:
 * 1. Replace floats with something less precise (maybe 16 bit)
 * 2. Row major vs column major -> if there is a nested loop then the inner change should be changing the most. Aim is to run allow memory continuosly
 * 3. Tiling 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <xmmintrin.h>
#include <string.h>
#include <mpi.h>

#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
  int number_of_ranks;
  int rank_id;
  int number_of_obstacles;
} t_param;

/* struct to hold the 'speed' arrays */
typedef struct
{
  float* restrict speed0;
  float* restrict speed1;
  float* restrict speed2;
  float* restrict speed3;
  float* restrict speed4;
  float* restrict speed5;
  float* restrict speed6;
  float* restrict speed7;
  float* restrict speed8;
} t_speed;

MPI_Request send_requests[6];
MPI_Request recv_requests[6];
int timestep_count = 0;
/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed* cells_ptr, t_speed* tmp_cells_ptr,
               char** obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/

float timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, char* obstacles);
int accelerate_flow(const t_param params, t_speed* cells, char* obstacles);
float propagate_rebound_and_collisions(const t_param params, t_speed* cells, t_speed* tmp_cells, char* obstacles);
int write_values(const t_param params, t_speed* cells, char* obstacles, float* av_vels);
int collision(const t_param params, t_speed* cells, t_speed* tmp_cells, char* obstacles);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed* cells_ptr, t_speed* tmp_cells_ptr,
             char** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, float av_velocity);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);
void swap(t_speed* cells, t_speed* cells2);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
    // MPI Setup
  int flag;               /* for checking whether MPI_Init() has been called */
  /* Initialize MPI */
  MPI_Init(&argc, &argv);

  /* check whether the initialisation was successful */
  MPI_Initialized(&flag);
  if ( flag != 1 ) {
    MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
  }

  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed *cells = malloc(sizeof(t_speed));  /* grid containing fluid densities */
  t_speed *tmp_cells = malloc(sizeof(t_speed));  /* grid indicating which cells are blocked */
  char    *obstacles = NULL;    
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;        /* structure to hold elapsed time */
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* Total/init time starts here: initialise our data structures and load values from file */
  gettimeofday(&timstr, NULL);
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic=tot_tic;
  initialise(paramfile, obstaclefile, &params, cells, tmp_cells, &obstacles, &av_vels);

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;
  
  float final_av_velocity;
  for (int tt = 0; tt < params.maxIters; tt++)
  {
    final_av_velocity = timestep(params, cells, tmp_cells, obstacles);
    av_vels[tt] = final_av_velocity;

#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }

  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic=comp_toc;

  // Collate data from ranks here 

  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;

  /* write final values and free memory */
  if (params.rank_id == 0) {
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, final_av_velocity));
    printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
    printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
    printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
    printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
  }
  write_values(params, cells, obstacles, av_vels);
  finalise(&params, cells, tmp_cells, &obstacles, &av_vels);

  /* finialise the MPI enviroment */
  MPI_Finalize();

  return EXIT_SUCCESS;
}

float timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, char* obstacles)
{
  // Only the final rank needs to accelerate_flow
  if (params.rank_id == params.number_of_ranks - 1) {
    accelerate_flow(params, cells, obstacles);
  }
  float time_step_solution = propagate_rebound_and_collisions(params, cells, tmp_cells, obstacles);
  swap(cells, tmp_cells);
  timestep_count++;
  return time_step_solution;
}

int accelerate_flow(const t_param params, t_speed* cells, char* restrict obstacles)
{
  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  int jj = params.ny - 2;
  
  __assume_aligned(cells->speed0, 64);
  __assume_aligned(cells->speed1, 64);
  __assume_aligned(cells->speed2, 64);
  __assume_aligned(cells->speed3, 64);
  __assume_aligned(cells->speed4, 64);
  __assume_aligned(cells->speed5, 64);
  __assume_aligned(cells->speed6, 64);
  __assume_aligned(cells->speed7, 64);
  __assume_aligned(cells->speed8, 64);
  
  __assume(params.nx%128==0);

  // #pragma omp simd
  #pragma omp parallel for
  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj*params.nx]
        && (cells->speed3[ii + jj*params.nx] - w1) > 0.f
        && (cells->speed6[ii + jj*params.nx] - w2) > 0.f
        && (cells->speed7[ii + jj*params.nx] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells->speed1[ii + jj*params.nx] += w1;
      cells->speed5[ii + jj*params.nx] += w2;
      cells->speed8[ii + jj*params.nx] += w2;
      /* decrease 'west-side' densities */
      cells->speed3[ii + jj*params.nx] -= w1;
      cells->speed6[ii + jj*params.nx] -= w2;
      cells->speed7[ii + jj*params.nx] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

float propagate_rebound_and_collisions(const t_param params, t_speed* cells, t_speed* tmp_cells, char* restrict obstacles)
{
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float c_2sq = 2.f * c_sq; /* 2 times square of speed of sound */
  const float c_2cu = c_2sq * c_sq; /* 2 times cube of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */
  
  int   tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u = 0.f;          /* accumulated magnitudes of velocity for each cell */

  __assume_aligned(cells->speed0, 64);
  __assume_aligned(cells->speed1, 64);
  __assume_aligned(cells->speed2, 64);
  __assume_aligned(cells->speed3, 64);
  __assume_aligned(cells->speed4, 64);
  __assume_aligned(cells->speed5, 64);
  __assume_aligned(cells->speed6, 64);
  __assume_aligned(cells->speed7, 64);
  __assume_aligned(cells->speed8, 64);
  
  __assume_aligned(tmp_cells->speed0, 64);
  __assume_aligned(tmp_cells->speed1, 64);
  __assume_aligned(tmp_cells->speed2, 64);
  __assume_aligned(tmp_cells->speed3, 64);
  __assume_aligned(tmp_cells->speed4, 64);
  __assume_aligned(tmp_cells->speed5, 64);
  __assume_aligned(tmp_cells->speed6, 64);
  __assume_aligned(tmp_cells->speed7, 64);
  __assume_aligned(tmp_cells->speed8, 64);
  
  __assume_aligned(obstacles, 64);

  __assume(params.nx%2==0);
  __assume(params.ny%2==0);
  __assume(params.nx%4==0);
  __assume(params.ny%4==0);
  __assume(params.nx%8==0);
  __assume(params.ny%8==0);
  __assume(params.nx%16==0);
  __assume(params.ny%16==0);
  __assume(params.nx%32==0);
  __assume(params.ny%32==0);
  __assume(params.nx%64==0);
  __assume(params.ny%64==0);
  __assume(params.nx%128==0);
  __assume(params.ny%128==0);
  
  if (timestep_count == 0) {
    MPI_Isend(
        &cells->speed7[0],  // src data
        params.nx,  // amount of data to send
        MPI_FLOAT,  // data type
        params.rank_id == 0 ? params.number_of_ranks - 1 : params.rank_id - 1,  // Which rank to recieve from 
        7,
        MPI_COMM_WORLD,
        &send_requests[0]
    );
    MPI_Isend(
        &cells->speed4[0],  // src data
        params.nx,  // amount of data to send
        MPI_FLOAT,  // data type
        params.rank_id == 0 ? params.number_of_ranks - 1 : params.rank_id - 1,  // Which rank to recieve from 
        4,
        MPI_COMM_WORLD,
        &send_requests[1]
    );
    MPI_Isend(
        &cells->speed8[0],  // src data
        params.nx,  // amount of data to send
        MPI_FLOAT,  // data type
        params.rank_id == 0 ? params.number_of_ranks - 1 : params.rank_id - 1,  // Which rank to recieve from 
        8,
        MPI_COMM_WORLD,
        &send_requests[2]
    );
    
    MPI_Irecv(
        &cells->speed7[(params.ny) * params.nx],
        params.nx,  // amount of data
        MPI_FLOAT,  // data type
        (params.rank_id + 1) % params.number_of_ranks,  // Which rank to send to
        7,
        MPI_COMM_WORLD,
        &recv_requests[0]
    );
    MPI_Irecv(
        &cells->speed4[(params.ny) * params.nx],
        params.nx,  // amount of data
        MPI_FLOAT,  // data type
        (params.rank_id + 1) % params.number_of_ranks,  // Which rank to send to
        4,
        MPI_COMM_WORLD,
        &recv_requests[1]
    );
    MPI_Irecv(
        &cells->speed8[(params.ny) * params.nx],
        params.nx,  // amount of data
        MPI_FLOAT,  // data type
        (params.rank_id + 1) % params.number_of_ranks,  // Which rank to send to
        8,
        MPI_COMM_WORLD,
        &recv_requests[2]
    );
  
    MPI_Isend(
        &cells->speed6[(params.ny - 1) * params.nx],  // src data
        params.nx,  // amount of data to send
        MPI_FLOAT,  // data type
        (params.rank_id + 1) % params.number_of_ranks,  // Which rank to send to
        6,
        MPI_COMM_WORLD,
        &send_requests[3]
    );
    MPI_Isend(
        &cells->speed5[(params.ny - 1) * params.nx],  // src data
        params.nx,  // amount of data to send
        MPI_FLOAT,  // data type
        (params.rank_id + 1) % params.number_of_ranks,  // Which rank to send to
        5,
        MPI_COMM_WORLD,
        &send_requests[4]
    );
    MPI_Isend(
        &cells->speed2[(params.ny - 1) * params.nx],  // src data
        params.nx,  // amount of data to send
        MPI_FLOAT,  // data type
        (params.rank_id + 1) % params.number_of_ranks,  // Which rank to send to
        2,
        MPI_COMM_WORLD,
        &send_requests[5]
    );
    // and we have finished with the top buffer so we can get ready to recieve a new one
    MPI_Irecv(
        &cells->speed6[(params.ny + 1) * params.nx],
        params.nx,  // amount of data
        MPI_FLOAT,  // data type
        params.rank_id - 1 == -1 ? params.number_of_ranks - 1 : params.rank_id - 1,  // Which rank to recieve from 
        6,
        MPI_COMM_WORLD,
        &recv_requests[3]
    );
    MPI_Irecv(
        &cells->speed5[(params.ny + 1) * params.nx],
        params.nx,  // amount of data
        MPI_FLOAT,  // data type
        params.rank_id - 1 == -1 ? params.number_of_ranks - 1 : params.rank_id - 1,  // Which rank to recieve from 
        5,
        MPI_COMM_WORLD,
        &recv_requests[4]
    );
    MPI_Irecv(
        &cells->speed2[(params.ny + 1) * params.nx],
        params.nx,  // amount of data
        MPI_FLOAT,  // data type
        params.rank_id - 1 == -1 ? params.number_of_ranks - 1 : params.rank_id - 1,  // Which rank to recieve from 
        2,
        MPI_COMM_WORLD,
        &recv_requests[5]
    );
  }

  // This will be a thing
  // int start_row = params.rank_id % 2 == 0 ? 0 : params.ny - 1;
  // int end_row = params.rank_id % 2 == 0 ? params.ny - 1 : 0;
  
  // MPI_Waitall(3, send_requests, MPI_STATUSES_IGNORE);
  // MPI_Waitall(3, &recv_requests[3], MPI_STATUSES_IGNORE);

  // --- Calculate inner cells in parallel
  {
    #pragma omp parallel for schedule(static) reduction(+:tot_cells) reduction(+:tot_u)
    for (int jj = 1; jj < params.ny - 1; jj++)
    {    
      #pragma omp simd aligned(cells:64) aligned(tmp_cells:64) aligned(obstacles:64) reduction(+:tot_cells) reduction(+:tot_u)
      for (int ii = 0; ii < params.nx; ii++)
      {
        const int y_n = jj + 1;
        const int x_e = (ii + 1) % params.nx;
        const int y_s = (jj == 0) ? (params.ny + 1) : (jj - 1);
        const int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
        
        /* if the cell contains an obstacle */
        if (obstacles[jj*params.nx + ii])
        {
          /* called after propagate, so taking values from scratch space
          ** mirroring, and writing into main grid */
          tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx];
          tmp_cells->speed1[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx];
          tmp_cells->speed2[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx];
          tmp_cells->speed3[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx];
          tmp_cells->speed4[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx];
          tmp_cells->speed5[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx];
          tmp_cells->speed6[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx];
          tmp_cells->speed7[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx];
          tmp_cells->speed8[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx];
        } 
        // Deal with collisions
        else
        {

          /* compute local density total */
          float local_density = 0.f;
          local_density += cells->speed0[ii + jj*params.nx];
          local_density += cells->speed1[x_w + jj*params.nx];
          local_density += cells->speed2[ii + y_s*params.nx];
          local_density += cells->speed3[x_e + jj*params.nx];
          local_density += cells->speed4[ii + y_n*params.nx];
          local_density += cells->speed5[x_w + y_s*params.nx];
          local_density += cells->speed6[x_e + y_s*params.nx];
          local_density += cells->speed7[x_e + y_n*params.nx];
          local_density += cells->speed8[x_w + y_n*params.nx];

          /* compute x velocity component */
          const float u_x = (
              cells->speed1[x_w + jj*params.nx]
            - cells->speed3[x_e + jj*params.nx]
            + cells->speed5[x_w + y_s*params.nx]
            - cells->speed6[x_e + y_s*params.nx]
            - cells->speed7[x_e + y_n*params.nx]
            + cells->speed8[x_w + y_n*params.nx]
          ) / local_density;
          /* compute y velocity component */
          const float u_y = (
              cells->speed2[ii + y_s*params.nx]
            - cells->speed4[ii + y_n*params.nx]
            + cells->speed5[x_w + y_s*params.nx]
            + cells->speed6[x_e + y_s*params.nx]
            - cells->speed7[x_e + y_n*params.nx]
            - cells->speed8[x_w + y_n*params.nx]
            ) / local_density;

          /* velocity squared */
          const float u_sq = u_x * u_x + u_y * u_y;
          const float u_sqd2sq = u_sq / c_2sq;

          /* relaxation step */
          tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx]
                                                 + params.omega
                                                 * (
                                                    w0 * local_density * (1.f - u_sq / (c_2sq))
                                                    - cells->speed0[ii + jj*params.nx]
                                                  );
          
          tmp_cells->speed1[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx]
                                                 + params.omega
                                                 * (
                                                    w1 * local_density * (1.f + u_x / c_sq + (u_x * u_x) / c_2cu - u_sqd2sq)
                                                    - cells->speed1[x_w + jj*params.nx]
                                                  );
          
          tmp_cells->speed2[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx]
                                                 + params.omega
                                                 * (
                                                    w1 * local_density * (1.f + u_y / c_sq + (u_y * u_y) / c_2cu - u_sqd2sq)
                                                    - cells->speed2[ii + y_s*params.nx]
                                                  );
          
          tmp_cells->speed3[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx]
                                                 + params.omega
                                                 * (
                                                    w1 * local_density * (1.f - u_x / c_sq + (u_x * u_x) / c_2cu - u_sqd2sq) 
                                                    - cells->speed3[x_e + jj*params.nx]
                                                  );
          
          tmp_cells->speed4[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx]
                                                 + params.omega
                                                 * (
                                                    w1 * local_density * (1.f - u_y / c_sq + (u_y * u_y) / c_2cu - u_sqd2sq) 
                                                    - cells->speed4[ii + y_n*params.nx]
                                                  );
          
          tmp_cells->speed5[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx]
                                                 + params.omega
                                                 * (
                                                    w2 * local_density * (1.f + (u_x + u_y) / c_sq + ((u_x + u_y) * (u_x + u_y)) / c_2cu - u_sqd2sq)
                                                    - cells->speed5[x_w + y_s*params.nx]
                                                  );
          
          tmp_cells->speed6[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx]
                                                 + params.omega
                                                 * (
                                                    w2 * local_density * (1.f + (-u_x + u_y) / c_sq + ((-u_x + u_y) * (-u_x + u_y))  / c_2cu - u_sqd2sq)
                                                    - cells->speed6[x_e + y_s*params.nx]
                                                  );
          
          tmp_cells->speed7[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx]
                                                 + params.omega
                                                 * (
                                                    w2 * local_density * (1.f - (u_x + u_y) / c_sq + ((u_x + u_y) * (u_x + u_y))    / c_2cu - u_sqd2sq) 
                                                    - cells->speed7[x_e + y_n*params.nx]
                                                  );
          
          tmp_cells->speed8[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx]
                                                 + params.omega
                                                 * (
                                                    w2 * local_density * (1.f + (u_x - u_y) / c_sq + ((u_x - u_y) * (u_x - u_y)) / c_2cu - u_sqd2sq)
                                                    - cells->speed8[x_w + y_n*params.nx]
                                                  );
         
          tot_u += sqrtf(u_sq);
          ++tot_cells;
        }
      }
    }
  }
  
  MPI_Waitall(6, send_requests, MPI_STATUSES_IGNORE);
  MPI_Waitall(6, recv_requests, MPI_STATUSES_IGNORE);
  
  // --- Calculate row 0 --- //
    for (int jj = 0; jj < 1; jj++)
    {    
      // If we are doing the last row

      #pragma omp simd aligned(cells:64) aligned(tmp_cells:64) aligned(obstacles:64) reduction(+:tot_cells) reduction(+:tot_u)
      for (int ii = 0; ii < params.nx; ii++)
      {
        const int y_n = jj + 1;
        const int x_e = (ii + 1) % params.nx;
        const int y_s = (jj == 0) ? (params.ny + 1) : (jj - 1);
        const int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
        
        /* if the cell contains an obstacle */
        if (obstacles[jj*params.nx + ii])
        {
          /* called after propagate, so taking values from scratch space
          ** mirroring, and writing into main grid */
          tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx];
          tmp_cells->speed1[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx];
          tmp_cells->speed2[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx];
          tmp_cells->speed3[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx];
          tmp_cells->speed4[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx];
          tmp_cells->speed5[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx];
          tmp_cells->speed6[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx];
          tmp_cells->speed7[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx];
          tmp_cells->speed8[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx];
        } 
        // Deal with collisions
        else
        {

          /* compute local density total */
          float local_density = 0.f;
          local_density += cells->speed0[ii + jj*params.nx];
          local_density += cells->speed1[x_w + jj*params.nx];
          local_density += cells->speed2[ii + y_s*params.nx];
          local_density += cells->speed3[x_e + jj*params.nx];
          local_density += cells->speed4[ii + y_n*params.nx];
          local_density += cells->speed5[x_w + y_s*params.nx];
          local_density += cells->speed6[x_e + y_s*params.nx];
          local_density += cells->speed7[x_e + y_n*params.nx];
          local_density += cells->speed8[x_w + y_n*params.nx];

          /* compute x velocity component */
          const float u_x = (
              cells->speed1[x_w + jj*params.nx]
            - cells->speed3[x_e + jj*params.nx]
            + cells->speed5[x_w + y_s*params.nx]
            - cells->speed6[x_e + y_s*params.nx]
            - cells->speed7[x_e + y_n*params.nx]
            + cells->speed8[x_w + y_n*params.nx]
          ) / local_density;
          /* compute y velocity component */
          const float u_y = (
              cells->speed2[ii + y_s*params.nx]
            - cells->speed4[ii + y_n*params.nx]
            + cells->speed5[x_w + y_s*params.nx]
            + cells->speed6[x_e + y_s*params.nx]
            - cells->speed7[x_e + y_n*params.nx]
            - cells->speed8[x_w + y_n*params.nx]
            ) / local_density;

          /* velocity squared */
          const float u_sq = u_x * u_x + u_y * u_y;
          const float u_sqd2sq = u_sq / c_2sq;

          /* relaxation step */
          tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx]
                                                 + params.omega
                                                 * (
                                                    w0 * local_density * (1.f - u_sq / (c_2sq))
                                                    - cells->speed0[ii + jj*params.nx]
                                                  );
          
          tmp_cells->speed1[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx]
                                                 + params.omega
                                                 * (
                                                    w1 * local_density * (1.f + u_x / c_sq + (u_x * u_x) / c_2cu - u_sqd2sq)
                                                    - cells->speed1[x_w + jj*params.nx]
                                                  );
          
          tmp_cells->speed2[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx]
                                                 + params.omega
                                                 * (
                                                    w1 * local_density * (1.f + u_y / c_sq + (u_y * u_y) / c_2cu - u_sqd2sq)
                                                    - cells->speed2[ii + y_s*params.nx]
                                                  );
          
          tmp_cells->speed3[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx]
                                                 + params.omega
                                                 * (
                                                    w1 * local_density * (1.f - u_x / c_sq + (u_x * u_x) / c_2cu - u_sqd2sq) 
                                                    - cells->speed3[x_e + jj*params.nx]
                                                  );
          
          tmp_cells->speed4[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx]
                                                 + params.omega
                                                 * (
                                                    w1 * local_density * (1.f - u_y / c_sq + (u_y * u_y) / c_2cu - u_sqd2sq) 
                                                    - cells->speed4[ii + y_n*params.nx]
                                                  );
          
          tmp_cells->speed5[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx]
                                                 + params.omega
                                                 * (
                                                    w2 * local_density * (1.f + (u_x + u_y) / c_sq + ((u_x + u_y) * (u_x + u_y)) / c_2cu - u_sqd2sq)
                                                    - cells->speed5[x_w + y_s*params.nx]
                                                  );
          
          tmp_cells->speed6[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx]
                                                 + params.omega
                                                 * (
                                                    w2 * local_density * (1.f + (-u_x + u_y) / c_sq + ((-u_x + u_y) * (-u_x + u_y))  / c_2cu - u_sqd2sq)
                                                    - cells->speed6[x_e + y_s*params.nx]
                                                  );
          
          tmp_cells->speed7[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx]
                                                 + params.omega
                                                 * (
                                                    w2 * local_density * (1.f - (u_x + u_y) / c_sq + ((u_x + u_y) * (u_x + u_y))    / c_2cu - u_sqd2sq) 
                                                    - cells->speed7[x_e + y_n*params.nx]
                                                  );
          
          tmp_cells->speed8[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx]
                                                 + params.omega
                                                 * (
                                                    w2 * local_density * (1.f + (u_x - u_y) / c_sq + ((u_x - u_y) * (u_x - u_y)) / c_2cu - u_sqd2sq)
                                                    - cells->speed8[x_w + y_n*params.nx]
                                                  );
         
          tot_u += sqrtf(u_sq);
          ++tot_cells;
        }
      }
    }

  
  // --- Calculate last row --- //
    // If we are doing the last row
    // Make sure we have the required halo row
    // MPI_Waitall(3, recv_requests, MPI_STATUSES_IGNORE);
    // And that we have sent the current row
    // MPI_Waitall(3, &send_requests[3], MPI_STATUSES_IGNORE);
    // printf("Rank %d: Waited for top row stuff on timestep_count: %d\n", params.rank_id, timestep_count);
    for (int jj = params.ny - 1; jj < params.ny; jj++)
    {    
      // If we are doing the last row

      #pragma omp simd aligned(cells:64) aligned(tmp_cells:64) aligned(obstacles:64) reduction(+:tot_cells) reduction(+:tot_u)
      for (int ii = 0; ii < params.nx; ii++)
      {
        const int y_n = jj + 1;
        const int x_e = (ii + 1) % params.nx;
        const int y_s = (jj == 0) ? (params.ny + 1) : (jj - 1);
        const int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
        
        /* if the cell contains an obstacle */
        if (obstacles[jj*params.nx + ii])
        {
          /* called after propagate, so taking values from scratch space
          ** mirroring, and writing into main grid */
          tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx];
          tmp_cells->speed1[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx];
          tmp_cells->speed2[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx];
          tmp_cells->speed3[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx];
          tmp_cells->speed4[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx];
          tmp_cells->speed5[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx];
          tmp_cells->speed6[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx];
          tmp_cells->speed7[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx];
          tmp_cells->speed8[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx];
        } 
        // Deal with collisions
        else
        {

          /* compute local density total */
          float local_density = 0.f;
          local_density += cells->speed0[ii + jj*params.nx];
          local_density += cells->speed1[x_w + jj*params.nx];
          local_density += cells->speed2[ii + y_s*params.nx];
          local_density += cells->speed3[x_e + jj*params.nx];
          local_density += cells->speed4[ii + y_n*params.nx];
          local_density += cells->speed5[x_w + y_s*params.nx];
          local_density += cells->speed6[x_e + y_s*params.nx];
          local_density += cells->speed7[x_e + y_n*params.nx];
          local_density += cells->speed8[x_w + y_n*params.nx];

          /* compute x velocity component */
          const float u_x = (
              cells->speed1[x_w + jj*params.nx]
            - cells->speed3[x_e + jj*params.nx]
            + cells->speed5[x_w + y_s*params.nx]
            - cells->speed6[x_e + y_s*params.nx]
            - cells->speed7[x_e + y_n*params.nx]
            + cells->speed8[x_w + y_n*params.nx]
          ) / local_density;
          /* compute y velocity component */
          const float u_y = (
              cells->speed2[ii + y_s*params.nx]
            - cells->speed4[ii + y_n*params.nx]
            + cells->speed5[x_w + y_s*params.nx]
            + cells->speed6[x_e + y_s*params.nx]
            - cells->speed7[x_e + y_n*params.nx]
            - cells->speed8[x_w + y_n*params.nx]
            ) / local_density;

          /* velocity squared */
          const float u_sq = u_x * u_x + u_y * u_y;
          const float u_sqd2sq = u_sq / c_2sq;

          /* relaxation step */
          tmp_cells->speed0[ii + jj*params.nx] = cells->speed0[ii + jj*params.nx]
                                                 + params.omega
                                                 * (
                                                    w0 * local_density * (1.f - u_sq / (c_2sq))
                                                    - cells->speed0[ii + jj*params.nx]
                                                  );
          
          tmp_cells->speed1[ii + jj*params.nx] = cells->speed1[x_w + jj*params.nx]
                                                 + params.omega
                                                 * (
                                                    w1 * local_density * (1.f + u_x / c_sq + (u_x * u_x) / c_2cu - u_sqd2sq)
                                                    - cells->speed1[x_w + jj*params.nx]
                                                  );
          
          tmp_cells->speed2[ii + jj*params.nx] = cells->speed2[ii + y_s*params.nx]
                                                 + params.omega
                                                 * (
                                                    w1 * local_density * (1.f + u_y / c_sq + (u_y * u_y) / c_2cu - u_sqd2sq)
                                                    - cells->speed2[ii + y_s*params.nx]
                                                  );
          
          tmp_cells->speed3[ii + jj*params.nx] = cells->speed3[x_e + jj*params.nx]
                                                 + params.omega
                                                 * (
                                                    w1 * local_density * (1.f - u_x / c_sq + (u_x * u_x) / c_2cu - u_sqd2sq) 
                                                    - cells->speed3[x_e + jj*params.nx]
                                                  );
          
          tmp_cells->speed4[ii + jj*params.nx] = cells->speed4[ii + y_n*params.nx]
                                                 + params.omega
                                                 * (
                                                    w1 * local_density * (1.f - u_y / c_sq + (u_y * u_y) / c_2cu - u_sqd2sq) 
                                                    - cells->speed4[ii + y_n*params.nx]
                                                  );
          
          tmp_cells->speed5[ii + jj*params.nx] = cells->speed5[x_w + y_s*params.nx]
                                                 + params.omega
                                                 * (
                                                    w2 * local_density * (1.f + (u_x + u_y) / c_sq + ((u_x + u_y) * (u_x + u_y)) / c_2cu - u_sqd2sq)
                                                    - cells->speed5[x_w + y_s*params.nx]
                                                  );
          
          tmp_cells->speed6[ii + jj*params.nx] = cells->speed6[x_e + y_s*params.nx]
                                                 + params.omega
                                                 * (
                                                    w2 * local_density * (1.f + (-u_x + u_y) / c_sq + ((-u_x + u_y) * (-u_x + u_y))  / c_2cu - u_sqd2sq)
                                                    - cells->speed6[x_e + y_s*params.nx]
                                                  );
          
          tmp_cells->speed7[ii + jj*params.nx] = cells->speed7[x_e + y_n*params.nx]
                                                 + params.omega
                                                 * (
                                                    w2 * local_density * (1.f - (u_x + u_y) / c_sq + ((u_x + u_y) * (u_x + u_y))    / c_2cu - u_sqd2sq) 
                                                    - cells->speed7[x_e + y_n*params.nx]
                                                  );
          
          tmp_cells->speed8[ii + jj*params.nx] = cells->speed8[x_w + y_n*params.nx]
                                                 + params.omega
                                                 * (
                                                    w2 * local_density * (1.f + (u_x - u_y) / c_sq + ((u_x - u_y) * (u_x - u_y)) / c_2cu - u_sqd2sq)
                                                    - cells->speed8[x_w + y_n*params.nx]
                                                  );
         
          tot_u += sqrtf(u_sq);
          ++tot_cells;
        }
      }
    }
  
    MPI_Isend(
        &cells->speed7[0],  // src data
        params.nx,  // amount of data to send
        MPI_FLOAT,  // data type
        params.rank_id == 0 ? params.number_of_ranks - 1 : params.rank_id - 1,  // Which rank to recieve from 
        7,
        MPI_COMM_WORLD,
        &send_requests[0]
    );
    MPI_Isend(
        &cells->speed4[0],  // src data
        params.nx,  // amount of data to send
        MPI_FLOAT,  // data type
        params.rank_id == 0 ? params.number_of_ranks - 1 : params.rank_id - 1,  // Which rank to recieve from 
        4,
        MPI_COMM_WORLD,
        &send_requests[1]
    );
    MPI_Isend(
        &cells->speed8[0],  // src data
        params.nx,  // amount of data to send
        MPI_FLOAT,  // data type
        params.rank_id == 0 ? params.number_of_ranks - 1 : params.rank_id - 1,  // Which rank to recieve from 
        8,
        MPI_COMM_WORLD,
        &send_requests[2]
    );
    
    MPI_Irecv(
        &cells->speed7[(params.ny) * params.nx],
        params.nx,  // amount of data
        MPI_FLOAT,  // data type
        (params.rank_id + 1) % params.number_of_ranks,  // Which rank to send to
        7,
        MPI_COMM_WORLD,
        &recv_requests[0]
    );
    MPI_Irecv(
        &cells->speed4[(params.ny) * params.nx],
        params.nx,  // amount of data
        MPI_FLOAT,  // data type
        (params.rank_id + 1) % params.number_of_ranks,  // Which rank to send to
        4,
        MPI_COMM_WORLD,
        &recv_requests[1]
    );
    MPI_Irecv(
        &cells->speed8[(params.ny) * params.nx],
        params.nx,  // amount of data
        MPI_FLOAT,  // data type
        (params.rank_id + 1) % params.number_of_ranks,  // Which rank to send to
        8,
        MPI_COMM_WORLD,
        &recv_requests[2]
    );
  
    MPI_Isend(
        &cells->speed6[(params.ny - 1) * params.nx],  // src data
        params.nx,  // amount of data to send
        MPI_FLOAT,  // data type
        (params.rank_id + 1) % params.number_of_ranks,  // Which rank to send to
        6,
        MPI_COMM_WORLD,
        &send_requests[3]
    );
    MPI_Isend(
        &cells->speed5[(params.ny - 1) * params.nx],  // src data
        params.nx,  // amount of data to send
        MPI_FLOAT,  // data type
        (params.rank_id + 1) % params.number_of_ranks,  // Which rank to send to
        5,
        MPI_COMM_WORLD,
        &send_requests[4]
    );
    MPI_Isend(
        &cells->speed2[(params.ny - 1) * params.nx],  // src data
        params.nx,  // amount of data to send
        MPI_FLOAT,  // data type
        (params.rank_id + 1) % params.number_of_ranks,  // Which rank to send to
        2,
        MPI_COMM_WORLD,
        &send_requests[5]
    );
    // and we have finished with the top buffer so we can get ready to recieve a new one
    MPI_Irecv(
        &cells->speed6[(params.ny + 1) * params.nx],
        params.nx,  // amount of data
        MPI_FLOAT,  // data type
        params.rank_id - 1 == -1 ? params.number_of_ranks - 1 : params.rank_id - 1,  // Which rank to recieve from 
        6,
        MPI_COMM_WORLD,
        &recv_requests[3]
    );
    MPI_Irecv(
        &cells->speed5[(params.ny + 1) * params.nx],
        params.nx,  // amount of data
        MPI_FLOAT,  // data type
        params.rank_id - 1 == -1 ? params.number_of_ranks - 1 : params.rank_id - 1,  // Which rank to recieve from 
        5,
        MPI_COMM_WORLD,
        &recv_requests[4]
    );
    MPI_Irecv(
        &cells->speed2[(params.ny + 1) * params.nx],
        params.nx,  // amount of data
        MPI_FLOAT,  // data type
        params.rank_id - 1 == -1 ? params.number_of_ranks - 1 : params.rank_id - 1,  // Which rank to recieve from 
        2,
        MPI_COMM_WORLD,
        &recv_requests[5]
    );

  float all_grids_tot_u;
  int all_grids_tot_cells;
  MPI_Allreduce(&tot_u, &all_grids_tot_u, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  // TODO just store the number of cells
  MPI_Allreduce(&tot_cells, &all_grids_tot_cells, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  return all_grids_tot_u/ (float)all_grids_tot_cells;
}



int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed* cells_ptr, t_speed* tmp_cells_ptr,
               char** obstacles_ptr, float** av_vels_ptr)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* Set rank */
  int rank_id;               /* 'rank' of process among it's cohort */ 
  int number_of_ranks;               /* size of cohort, i.e. num processes started */
  MPI_Comm_rank(MPI_COMM_WORLD,&rank_id);
  params->rank_id = rank_id;
  MPI_Comm_size(MPI_COMM_WORLD,&number_of_ranks);
  params->number_of_ranks = number_of_ranks;

  // Now that the grid is going to be split into different regions each grid
  // will become a subset of the main grid.

  // Split the grid into n (number of ranks) grids.
  // TODO For now assumed grid is divisible by 4
  params->ny /= params->number_of_ranks;
  int number_of_cells = params->nx * params->ny;

  // Two halo regions will be added to the end of the cells grid
  int number_of_cells_with_halo = params->nx * (params->ny + 2);

  int rank_start_row = params->ny * params->rank_id; 
  int rank_end_row = params->ny * (params->rank_id + 1) - 1;

  /* main grid */
  cells_ptr->speed0 = _mm_malloc(number_of_cells_with_halo * sizeof(float), 64);
  cells_ptr->speed1 = _mm_malloc(number_of_cells_with_halo * sizeof(float), 64);
  cells_ptr->speed2 = _mm_malloc(number_of_cells_with_halo * sizeof(float), 64);
  cells_ptr->speed3 = _mm_malloc(number_of_cells_with_halo * sizeof(float), 64);
  cells_ptr->speed4 = _mm_malloc(number_of_cells_with_halo * sizeof(float), 64);
  cells_ptr->speed5 = _mm_malloc(number_of_cells_with_halo * sizeof(float), 64);
  cells_ptr->speed6 = _mm_malloc(number_of_cells_with_halo * sizeof(float), 64);
  cells_ptr->speed7 = _mm_malloc(number_of_cells_with_halo * sizeof(float), 64);
  cells_ptr->speed8 = _mm_malloc(number_of_cells_with_halo * sizeof(float), 64);

  if (cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  tmp_cells_ptr->speed0 = _mm_malloc(number_of_cells_with_halo * sizeof(float), 64);
  tmp_cells_ptr->speed1 = _mm_malloc(number_of_cells_with_halo * sizeof(float), 64);
  tmp_cells_ptr->speed2 = _mm_malloc(number_of_cells_with_halo * sizeof(float), 64);
  tmp_cells_ptr->speed3 = _mm_malloc(number_of_cells_with_halo * sizeof(float), 64);
  tmp_cells_ptr->speed4 = _mm_malloc(number_of_cells_with_halo * sizeof(float), 64);
  tmp_cells_ptr->speed5 = _mm_malloc(number_of_cells_with_halo * sizeof(float), 64);
  tmp_cells_ptr->speed6 = _mm_malloc(number_of_cells_with_halo * sizeof(float), 64);
  tmp_cells_ptr->speed7 = _mm_malloc(number_of_cells_with_halo * sizeof(float), 64);
  tmp_cells_ptr->speed8 = _mm_malloc(number_of_cells_with_halo * sizeof(float), 64);
  if (tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = _mm_malloc(sizeof(char) * (number_of_cells), 64);
  // *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  #pragma omp parallel for
  for (int jj = 0; jj < params->ny + 2; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      cells_ptr->speed0[ii + jj*params->nx] = w0;
      /* axis directions */
      cells_ptr->speed1[ii + jj*params->nx] = w1;
      cells_ptr->speed2[ii + jj*params->nx] = w1;
      cells_ptr->speed3[ii + jj*params->nx] = w1;
      cells_ptr->speed4[ii + jj*params->nx] = w1;
      /* diagonals */
      cells_ptr->speed5[ii + jj*params->nx] = w2;
      cells_ptr->speed6[ii + jj*params->nx] = w2;
      cells_ptr->speed7[ii + jj*params->nx] = w2;
      cells_ptr->speed8[ii + jj*params->nx] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  #pragma omp parallel for
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    if (yy >= rank_start_row && yy <= rank_end_row) {
      /* assign to array */
      int local_row = yy - rank_start_row;
      (*obstacles_ptr)[xx + local_row*params->nx] = blocked;
    }
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed* cells_ptr, t_speed* tmp_cells_ptr,
             char** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  _mm_free(cells_ptr->speed0);
  _mm_free(cells_ptr->speed1);
  _mm_free(cells_ptr->speed2);
  _mm_free(cells_ptr->speed3);
  _mm_free(cells_ptr->speed4);
  _mm_free(cells_ptr->speed5);
  _mm_free(cells_ptr->speed6);
  _mm_free(cells_ptr->speed7);
  _mm_free(cells_ptr->speed8);
  free(cells_ptr);
  cells_ptr = NULL;

  _mm_free(tmp_cells_ptr->speed0);
  _mm_free(tmp_cells_ptr->speed1);
  _mm_free(tmp_cells_ptr->speed2);
  _mm_free(tmp_cells_ptr->speed3);
  _mm_free(tmp_cells_ptr->speed4);
  _mm_free(tmp_cells_ptr->speed5);
  _mm_free(tmp_cells_ptr->speed6);
  _mm_free(tmp_cells_ptr->speed7);
  _mm_free(tmp_cells_ptr->speed8);
  free(tmp_cells_ptr);
  tmp_cells_ptr = NULL;

  _mm_free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, float av_velocity)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      
      total += cells->speed0[ii + jj*params.nx];
      total += cells->speed1[ii + jj*params.nx];
      total += cells->speed2[ii + jj*params.nx];
      total += cells->speed3[ii + jj*params.nx];
      total += cells->speed4[ii + jj*params.nx];
      total += cells->speed5[ii + jj*params.nx];
      total += cells->speed6[ii + jj*params.nx];
      total += cells->speed7[ii + jj*params.nx];
      total += cells->speed8[ii + jj*params.nx];
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells, char* obstacles, float* av_vels)
{
  MPI_File fh;
  MPI_Status status;

  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  const int output_line_length = 140;
  char buf[output_line_length * params.nx * params.ny];
  MPI_File_open(MPI_COMM_SELF, FINALSTATEFILE, MPI_MODE_CREATE | MPI_MODE_WRONLY,MPI_INFO_NULL, &fh);

  if (fh == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.f;
      
        local_density += cells->speed0[ii + jj*params.nx];
        local_density += cells->speed1[ii + jj*params.nx];
        local_density += cells->speed2[ii + jj*params.nx];
        local_density += cells->speed3[ii + jj*params.nx];
        local_density += cells->speed4[ii + jj*params.nx];
        local_density += cells->speed5[ii + jj*params.nx];
        local_density += cells->speed6[ii + jj*params.nx];
        local_density += cells->speed7[ii + jj*params.nx];
        local_density += cells->speed8[ii + jj*params.nx];

        /* compute x velocity component */
        u_x = (  cells->speed1[ii + jj*params.nx]
               + cells->speed5[ii + jj*params.nx]
               + cells->speed8[ii + jj*params.nx]
               - (cells->speed3[ii + jj*params.nx]
                  + cells->speed6[ii + jj*params.nx]
                  + cells->speed7[ii + jj*params.nx]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells->speed2[ii + jj*params.nx]
               + cells->speed5[ii + jj*params.nx]
               + cells->speed6[ii + jj*params.nx]
               - (cells->speed4[ii + jj*params.nx]
                  + cells->speed7[ii + jj*params.nx]
                  + cells->speed8[ii + jj*params.nx]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      sprintf(buf + strlen(buf), "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj + (params.rank_id * params.ny), u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  const int offsets[params.number_of_ranks];
  const int bufferLength = strlen(buf);
  MPI_Allgather(&bufferLength, 1, MPI_INT, &offsets, 1, MPI_INT, MPI_COMM_WORLD);

  int offset = 0;
  for (int i = 0; i < params.rank_id; i++) {
    offset += offsets[i];
  }
  printf("%d's offset is %d\n", params.rank_id, offset);

  MPI_File_seek(fh, offset, MPI_SEEK_SET);
  printf("Writing to file");
  MPI_File_write(fh,buf,strlen(buf), MPI_CHAR,&status);
  MPI_File_close(&fh);

  if (params.rank_id == 0) {
    FILE* fp;                     /* file pointer */
    fp = fopen(AVVELSFILE, "w");

    if (fp == NULL)
    {
      die("could not open file output file", __LINE__, __FILE__);
    }

    for (int ii = 0; ii < params.maxIters; ii++)
    {
      fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
    }

    fclose(fp);
  }

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}

// void swap(t_speed* cells, t_speed* cells2) {
//   /* Swaps pointers of cells */
//   t_speed* tmp = cells;
//   cells = cells2;
//   cells2 = tmp;
// }

void swap(t_speed *a, t_speed *b) {
  t_speed tmp = *a;
  *a = *b;
  *b = tmp;
}
