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
#include <mpi.h>


#define NSPEEDS         9
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
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               char** obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/

float timestep(const t_param params, t_speed** cells, t_speed** tmp_cells, char* obstacles);
int accelerate_flow(const t_param params, t_speed* cells, char* obstacles);
float propagate_rebound_and_collisions(const t_param params, t_speed* cells, t_speed* tmp_cells, char* obstacles);
int write_values(const t_param params, t_speed* cells, char* obstacles, float* av_vels);
int collision(const t_param params, t_speed* cells, t_speed* tmp_cells, char* obstacles);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             char** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, char* obstacles, float av_velocity);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);
void swap(t_speed** cells, t_speed** cells2);
void debug_print_cells(t_speed* cells, const t_param params);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{

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
  t_speed* cells  = NULL;    /* grid containing fluid densities */
  t_speed* tmp_cells = NULL;    /* scratch space */
  char*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */

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

  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);

  /* iterate for maxIters timesteps */
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  float final_av_velocity;

  // for (int tt = 0; tt < params.maxIters; tt++)
  for (int tt = 0; tt < 1; tt++)
  {
    final_av_velocity = timestep(params, &cells, &tmp_cells, obstacles);
    av_vels[tt] = final_av_velocity;
// #ifdef DEBUG
    if (params.rank_id == 0) {
      printf("==timestep: %d==\n", tt);
      printf("av velocity: %.12E\n", av_vels[tt]);
      printf("tot density: %.12E\n", total_density(params, cells));
    }
// #endif
  }

  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  /* write final values and free memory */
  if (params.rank_id == 0) {
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles, final_av_velocity));
    printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
    printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
    printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
    write_values(params, cells, obstacles, av_vels);
  }
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

  /* finialise the MPI enviroment */
  MPI_Finalize();

  return EXIT_SUCCESS;
}

float timestep(const t_param params, t_speed** cells, t_speed** tmp_cells, char* obstacles)
{
  accelerate_flow(params, *cells, obstacles);
  float time_step_solution = propagate_rebound_and_collisions(params, *cells, *tmp_cells, obstacles);
  swap(cells, tmp_cells);
  // collision(params, cells, tmp_cells, obstacles);
  return time_step_solution;
}

int accelerate_flow(const t_param params, t_speed* cells, char* obstacles)
{
  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  int jj = params.ny - 2 + 1;

  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj*params.nx]
        && (cells[ii + jj*params.nx].speeds[3] - w1) > 0.f
        && (cells[ii + jj*params.nx].speeds[6] - w2) > 0.f
        && (cells[ii + jj*params.nx].speeds[7] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells[ii + jj*params.nx].speeds[1] += w1;
      cells[ii + jj*params.nx].speeds[5] += w2;
      cells[ii + jj*params.nx].speeds[8] += w2;
      /* decrease 'west-side' densities */
      cells[ii + jj*params.nx].speeds[3] -= w1;
      cells[ii + jj*params.nx].speeds[6] -= w2;
      cells[ii + jj*params.nx].speeds[7] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

float propagate_rebound_and_collisions(const t_param params, t_speed* restrict cells, t_speed* restrict tmp_cells, char* obstacles)
{

  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float c_2sq = 2.f * c_sq; /* 2 times square of speed of sound */
  const float c_2cu = c_2sq * c_sq; /* 2 times cube of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */
  
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u = 0.f;          /* accumulated magnitudes of velocity for each cell */

  // Create mpi datatype
  MPI_Datatype MPI_T_SPEEDS;
  MPI_Aint displacements[1];
  const int nr_blocks = 1;
  int blocklengths[1] = {NSPEEDS};
  MPI_Datatype oldtypes[1] = {MPI_FLOAT};

  displacements[0] = offsetof(t_speed, speeds);

  MPI_Type_create_struct(nr_blocks, blocklengths, displacements,
                   oldtypes, &MPI_T_SPEEDS);
  MPI_Type_commit(&MPI_T_SPEEDS);
  
  MPI_Status status;
 
  int send_rank_id = params.rank_id - 1;
  if (send_rank_id == -1) {
    send_rank_id = params.number_of_ranks - 1;
  }
  int rec_rank_id = (params.rank_id + 1) % params.number_of_ranks;
  
  // Send the first non halo row
  MPI_Sendrecv(
      &cells[params.nx],  // src data
      params.nx,  // amount of data to send
      MPI_T_SPEEDS,  // data type
      send_rank_id,  // Which rank to send to
      0,
      
      // Recieve and store in the bottom halo row
      &cells[(params.ny + 1) * params.nx],
      params.nx,  // amount of data
      MPI_T_SPEEDS,  // data type
      rec_rank_id,  // Which rank to recieve from 
      0,

      //
      MPI_COMM_WORLD,
      &status
  );
  
  send_rank_id = (params.rank_id + 1) % params.number_of_ranks;
  rec_rank_id = params.rank_id - 1;
  if (rec_rank_id == -1) {
    rec_rank_id = params.number_of_ranks - 1;
  }
  
  // Send the last non halo row
  MPI_Sendrecv(
      &(cells[(params.ny) * params.nx]),
      params.nx,  // amount of data to send
      MPI_T_SPEEDS,  // data type
      send_rank_id,  // Which rank to send to
      0,
      
      // Recieve and store in halo top row
      &(cells[0]),
      params.nx,  // amount of data
      MPI_T_SPEEDS,  // data type
      rec_rank_id,  // Which rank to recieve from 
      0,

      //
      MPI_COMM_WORLD,
      &status
  );
  printf("nx = %d, ny = %d\n", params.nx, params.ny);
  printf("%d is sending to %d and recieving from %d\n", params.rank_id, send_rank_id, rec_rank_id);
  if (params.rank_id == 0) {
    printf("rec value: %.12f\n", cells[0].speeds[0]);
    printf("rec value: %.12f\n", cells[0].speeds[8]);
  } else if (params.rank_id == 3) {
    printf("sent value: %.12f\n", cells[(params.ny) * params.nx].speeds[0]);
    printf("sent value: %.12f\n", cells[(params.ny) * params.nx].speeds[8]);
  }


  float speed_1_sum = 0;
  float all_speed_1_sum;
  for (int jj = 1; jj < params.ny + 1; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      speed_1_sum += cells[(jj * params.nx) + ii].speeds[3];
    }
  }
  MPI_Allreduce(&speed_1_sum, &all_speed_1_sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  printf("speed3 sum: %.12f\n", all_speed_1_sum);
  
 
  debug_print_cells(cells, params);

  if (params.rank_id == 3) {
    printf("-1, 0, 0, 3: %.12f\n", cells[(params.ny + 1) * params.nx].speeds[0]);
  }
  if (params.rank_id == 0) {
    printf("-1, 0, 0: %.12f\n", cells[params.nx].speeds[0]);
    printf("-1, 0, 0: %.12f\n", cells[0].speeds[0]);
    printf("-1, 0, 0: %.12f\n", cells[(params.ny + 1) * params.nx].speeds[0]);
  }
  
  /* loop over _all_ cells */
  if (params.rank_id == 0)
  {
    // #pragma omp parallel for simd collapse(2)
    for (int jj = 2; jj < 3; jj++)
    {
      for (int ii = 1; ii < 2; ii++)
      {
        printf("Cell values speed 0: %.12f\n", cells[ii + jj * params.nx].speeds[0]);
        printf("Cell values speed 1: %.12f\n", cells[ii + jj * params.nx].speeds[1]);
        printf("Cell values speed 2: %.12f\n", cells[ii + jj * params.nx].speeds[2]);
        printf("Cell values speed 3: %.12f\n", cells[ii + jj * params.nx].speeds[3]);
        printf("Cell values speed 4: %.12f\n", cells[ii + jj * params.nx].speeds[4]);
        printf("Cell values speed 5: %.12f\n", cells[ii + jj * params.nx].speeds[5]);
        printf("Cell values speed 6: %.12f\n", cells[ii + jj * params.nx].speeds[6]);
        printf("Cell values speed 7: %.12f\n", cells[ii + jj * params.nx].speeds[7]);
        printf("Cell values speed 8: %.12f\n", cells[ii + jj * params.nx].speeds[8]);
        const int y_n = jj + 1;
        const int x_e = (ii + 1) & (params.nx - 1);
        const int y_s = jj - 1;
        const int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
        
        /* if the cell contains an obstacle */
        if (obstacles[jj*params.nx + ii])
        {
          /* called after propagate, so taking values from scratch space
          ** mirroring, and writing into main grid */
          tmp_cells[ii + jj*params.nx].speeds[0] = cells[ii + jj*params.nx].speeds[0];
          tmp_cells[ii + jj*params.nx].speeds[1] = cells[x_e + jj*params.nx].speeds[3];
          tmp_cells[ii + jj*params.nx].speeds[2] = cells[ii + y_n*params.nx].speeds[4];
          tmp_cells[ii + jj*params.nx].speeds[3] = cells[x_w + jj*params.nx].speeds[1];
          tmp_cells[ii + jj*params.nx].speeds[4] = cells[ii + y_s*params.nx].speeds[2];
          tmp_cells[ii + jj*params.nx].speeds[5] = cells[x_e + y_n*params.nx].speeds[7];
          tmp_cells[ii + jj*params.nx].speeds[6] = cells[x_w + y_n*params.nx].speeds[8];
          tmp_cells[ii + jj*params.nx].speeds[7] = cells[x_w + y_s*params.nx].speeds[5];
          tmp_cells[ii + jj*params.nx].speeds[8] = cells[x_e + y_s*params.nx].speeds[6];
        } 
        // Deal with collisions
        else
        {
          // printf("value: %.32f\n", cells[x_w + jj*params.nx].speeds[1]);
          // printf("value: %.32f\n", cells[x_e + jj*params.nx].speeds[3]);
          // printf("value: %.32f\n", cells[x_w + y_s*params.nx].speeds[5]);
          // printf("value: %.32f\n", cells[x_e + y_s*params.nx].speeds[6]);
          // printf("value: %.32f\n", cells[x_e + y_n*params.nx].speeds[7]);
          // printf("value: %.32f\n", cells[x_w + y_n*params.nx].speeds[8]);

          // Propogate
          tmp_cells[ii + jj*params.nx].speeds[0] = cells[ii + jj*params.nx].speeds[0]; /* central cell, no movement */
          tmp_cells[ii + jj*params.nx].speeds[1] = cells[x_w + jj*params.nx].speeds[1]; /* east */
          tmp_cells[ii + jj*params.nx].speeds[2] = cells[ii + y_s*params.nx].speeds[2]; /* north */
          tmp_cells[ii + jj*params.nx].speeds[3] = cells[x_e + jj*params.nx].speeds[3]; /* west */
          tmp_cells[ii + jj*params.nx].speeds[4] = cells[ii + y_n*params.nx].speeds[4]; /* south */
          tmp_cells[ii + jj*params.nx].speeds[5] = cells[x_w + y_s*params.nx].speeds[5]; /* north-east */
          tmp_cells[ii + jj*params.nx].speeds[6] = cells[x_e + y_s*params.nx].speeds[6]; /* north-west */
          tmp_cells[ii + jj*params.nx].speeds[7] = cells[x_e + y_n*params.nx].speeds[7]; /* south-west */
          tmp_cells[ii + jj*params.nx].speeds[8] = cells[x_w + y_n*params.nx].speeds[8]; /* south-east */

          /* compute local density total */
          float local_density = 0.f;

          for (int kk = 0; kk < NSPEEDS; kk++)
          {
            local_density += tmp_cells[ii + jj*params.nx].speeds[kk];
          }

          /* compute x velocity component */
          const float u_x_pre_div = (
              cells[x_w + jj*params.nx].speeds[1]
            - cells[x_e + jj*params.nx].speeds[3]
            + cells[x_w + y_s*params.nx].speeds[5]
            - cells[x_e + y_s*params.nx].speeds[6]
            - cells[x_e + y_n*params.nx].speeds[7]
            + cells[x_w + y_n*params.nx].speeds[8]
          );

          const float left = (cells[x_w + jj*params.nx].speeds[1] - cells[x_e + jj*params.nx].speeds[3]);
          const float right = (cells[x_w + y_s*params.nx].speeds[5] - cells[x_e + y_s*params.nx].speeds[6]);
          const float leftandright = (cells[x_w + jj*params.nx].speeds[1] - cells[x_e + jj*params.nx].speeds[3]) + (cells[x_w + y_s*params.nx].speeds[5] - cells[x_e + y_s*params.nx].speeds[6]);
          printf("test value right : %.32f\n",  right);
          printf("test value left: %.32f\n",  left);
          printf("test value left and right: %.32f\n",  leftandright);
          printf("test value: %.32f\n",  left + right);

          const float u_x = u_x_pre_div / local_density;
          printf("%d, %d\n", x_w, jj);
          printf("%d, %d\n", x_e, jj);
          printf("%d, %d\n", x_w, y_s);
          printf("%d, %d\n", x_e, y_s);
          printf("%d, %d\n", x_e, y_n);
          printf("%d, %d\n", x_w, y_n);
          
          printf("local_density: %.32f\n", local_density);
          printf("value: %.32f\n", cells[x_w + jj*params.nx].speeds[1]);
          printf("value: %.32f\n", cells[x_e + jj*params.nx].speeds[3]);
          printf("value: %.32f\n", cells[x_w + y_s*params.nx].speeds[5]);
          printf("value: %.32f\n", cells[x_e + y_s*params.nx].speeds[6]);
          printf("value: %.32f\n", cells[x_e + y_n*params.nx].speeds[7]);
          printf("value: %.32f\n", cells[x_w + y_n*params.nx].speeds[8]);
          printf("u_x_pre_div: %.32f\n", u_x_pre_div);
          printf("u_x: %.32f\n", u_x);
          /* compute y velocity component */
          float u_y = (
              cells[ii + y_s*params.nx].speeds[2]
            - cells[ii + y_n*params.nx].speeds[4]
            + cells[x_w + y_s*params.nx].speeds[5]
            + cells[x_e + y_s*params.nx].speeds[6]
            - cells[x_e + y_n*params.nx].speeds[7]
            - cells[x_w + y_n*params.nx].speeds[8]
            ) / local_density;
          printf("u_y: %.32f\n", u_y);

          /* velocity squared */
          float u_sq = u_x * u_x + u_y * u_y;
          float u_sqd2sq = u_sq / c_2sq;

          /* directional velocity components */
          float u[NSPEEDS];
          u[1] =   u_x;        /* east */
          u[2] =         u_y;  /* north */
          u[3] = - u_x;        /* west */
          u[4] =       - u_y;  /* south */
          u[5] =   u_x + u_y;  /* north-east */
          u[6] = - u_x + u_y;  /* north-west */
          u[7] = - u_x - u_y;  /* south-west */
          u[8] =   u_x - u_y;  /* south-east */

          /* equilibrium densities */
          float d_equ[NSPEEDS];
          /* zero velocity density: weight w0 */
          d_equ[0] = w0 * local_density * (1.f - u_sq / (c_2sq));
          /* axis speeds: weight w1 */
          float w1ld = w1 * local_density;
          d_equ[1] = w1ld * (1.f + u[1] / c_sq + (u[1] * u[1]) / c_2cu - u_sqd2sq);
          d_equ[2] = w1ld * (1.f + u[2] / c_sq + (u[2] * u[2]) / c_2cu - u_sqd2sq);
          d_equ[3] = w1ld * (1.f + u[3] / c_sq + (u[3] * u[3]) / c_2cu - u_sqd2sq);
          d_equ[4] = w1ld * (1.f + u[4] / c_sq + (u[4] * u[4]) / c_2cu - u_sqd2sq);
          /* diagonal speeds: weight w2 */
          float w2ld = w2 * local_density;
          d_equ[5] = w2ld * (1.f + u[5] / c_sq + (u[5] * u[5]) / c_2cu - u_sqd2sq);
          d_equ[6] = w2ld * (1.f + u[6] / c_sq + (u[6] * u[6]) / c_2cu - u_sqd2sq);
          d_equ[7] = w2ld * (1.f + u[7] / c_sq + (u[7] * u[7]) / c_2cu - u_sqd2sq);
          d_equ[8] = w2ld * (1.f + u[8] / c_sq + (u[8] * u[8]) / c_2cu - u_sqd2sq);

          /* relaxation step */
          for (int kk = 0; kk < NSPEEDS; kk++)
          {
            tmp_cells[ii + jj*params.nx].speeds[kk] = tmp_cells[ii + jj*params.nx].speeds[kk]
                                                    + params.omega
                                                    * (d_equ[kk] - tmp_cells[ii + jj*params.nx].speeds[kk]);
          }
          
          tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
          ++tot_cells;
        }
      }
    }
  }

  float all_grids_tot_u;
  int all_grids_tot_cells;
  // TODO only do this reduce for 1 thing
  MPI_Allreduce(&tot_u, &all_grids_tot_u, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  // TODO just store the number of cells
  MPI_Allreduce(&tot_cells, &all_grids_tot_cells, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
  if (params.rank_id == 0) {
    printf("tot_cells: %d\n", all_grids_tot_cells);
    printf("tot_u: %.32f\n", all_grids_tot_u);
  }
  return all_grids_tot_u/ (float)all_grids_tot_cells;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               char** obstacles_ptr, float** av_vels_ptr)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */
  int rank_id;               /* 'rank' of process among it's cohort */ 
  int number_of_ranks;               /* size of cohort, i.e. num processes started */

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

  /* Set rank */
  MPI_Comm_rank(MPI_COMM_WORLD,&rank_id);
  params->rank_id = rank_id;
  MPI_Comm_size(MPI_COMM_WORLD,&number_of_ranks);
  params->number_of_ranks = number_of_ranks;

  /* Set grid size values */
  int full_grid_height = params->ny;
  params->ny /= params->number_of_ranks;

  /* Calculate actual row range of this rank */
  int start_row = params->ny * params->rank_id;
  int end_row = params->ny * params->rank_id + params->ny - 1;
  int number_of_cells = (params->ny + 2) * params->nx;

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

  /* main grid */
  *cells_ptr = (t_speed*)malloc(sizeof(t_speed) * number_of_cells);

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * number_of_cells);

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(char) * number_of_cells);

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < params->ny+2; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      (*cells_ptr)[ii + jj*params->nx].speeds[0] = w0;
      /* axis directions */
      (*cells_ptr)[ii + jj*params->nx].speeds[1] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[2] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[3] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[4] = w1;
      /* diagonals */
      (*cells_ptr)[ii + jj*params->nx].speeds[5] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[6] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[7] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[8] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny + 2; jj++)
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

    // if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* If in this rank's grid assigned to array */
    if (yy >= start_row && yy <= end_row) 
    {
      int index = yy - start_row;
      (*obstacles_ptr)[xx + (index + 1)*params->nx] = blocked;
      printf("Setting obstacle in position: %d\n", xx + (index+1)*params->nx);
    }
  }
  
  float speed_1_sum = 0;
  float all_speed_1_sum;
  int num_cells = 0;
  int all_num_cells;

  float value = (*cells_ptr)[params->nx].speeds[3];
  printf("first value = %.12f", value);
  for (int jj = 1; jj < params->ny + 1; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      if ((*cells_ptr)[(jj * params->nx) + ii].speeds[3] != value) {
        printf("Wrong value detected!!! %.12f", (*cells_ptr)[(jj * params->nx) + ii].speeds[3]);
      }
      speed_1_sum += (*cells_ptr)[(jj * params->nx) + ii].speeds[3];
      num_cells++;
    }
  }
  MPI_Allreduce(&speed_1_sum, &all_speed_1_sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&num_cells, &all_num_cells, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  printf("init speed3 sum: %.12f, in %d cells\n", all_speed_1_sum, all_num_cells);
  
  printf("Process %d of %d. My starting row is %d and my end row is %d.\n", params->rank_id, params->number_of_ranks, start_row, end_row);

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             char** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed* cells, char* obstacles, float av_velocity)
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
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[ii + jj*params.nx].speeds[kk];
      }
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells, char* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");
  // MPI_File *fp;
  // MPI_File_open(MPI_COMM_WORLD, FINALSTATEFILE, MPI_MODE_WRONLY, MPI_INFO_NULL, fp)

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 1; jj < params.ny + 1; jj++)
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

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj*params.nx].speeds[kk];
        }

        /* compute x velocity component */
        u_x = (cells[ii + jj*params.nx].speeds[1]
               + cells[ii + jj*params.nx].speeds[5]
               + cells[ii + jj*params.nx].speeds[8]
               - (cells[ii + jj*params.nx].speeds[3]
                  + cells[ii + jj*params.nx].speeds[6]
                  + cells[ii + jj*params.nx].speeds[7]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells[ii + jj*params.nx].speeds[2]
               + cells[ii + jj*params.nx].speeds[5]
               + cells[ii + jj*params.nx].speeds[6]
               - (cells[ii + jj*params.nx].speeds[4]
                  + cells[ii + jj*params.nx].speeds[7]
                  + cells[ii + jj*params.nx].speeds[8]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      // MPI_File_write_at(fp, MPI_Offset offset, ROMIO_CONST void *buf,
      //        int count, MPI_Datatype datatype, MPI_Status *status);
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii + (params.rank_id * params.ny), jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);
  // MPI_File_close(fp)

  // MPI_File_open(MPI_COMM_WORLD, AVVELSFILE, MPI_MODE_WRONLY, MPI_INFO_NULL, fp)
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

void swap(t_speed** cells, t_speed** cells2) {
  /* Swaps pointers of cells */
  t_speed *tmp = *cells;
  *cells = *cells2;
  *cells2 = tmp;
}

void debug_print_cells(t_speed* cells, const t_param params) {
  // FILE* fp = fopen("debug_cells_output.dat", "w");
  MPI_File fh;
  MPI_Status status;
 
  MPI_File_open(MPI_COMM_SELF, "test.txt",MPI_MODE_CREATE | MPI_MODE_WRONLY,MPI_INFO_NULL,&fh);
  // if (fp == NULL)
  // {
  //   die("could not open debug output file", __LINE__, __FILE__);
  // }

  for (int xx = 0; xx < params.nx; xx++)
  {
    for (int yy = 1; yy < params.ny + 1; yy++)
    {
      char buf[40];
      snprintf(buf, 40, "%d %d %.12E\n", xx, yy + (params.rank_id * params.ny), cells[yy * params.nx + xx].speeds[0]);
      MPI_File_write(fh,buf,40, MPI_CHAR,&status);
    }
  }
  MPI_File_close(&fh);
}