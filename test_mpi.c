#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <stddef.h>

#define NSPEEDS 9
#define NX 4
#define NY 4

typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

int initialise(t_speed **cells, int rank_id) {
  *cells = (t_speed*)malloc(sizeof(t_speed) * NX * NY);
  for (int i = 0; i < NX; i++) {
    for (int j = 0; j < NX; j++) {
      for (int k = 0; k < NSPEEDS; k++) {
        (*cells)[i + j*NX].speeds[k] = rank_id;
      }
    }
  }
}
  
int main(int argc, char* argv[])
{
  int flag;               /* for checking whether MPI_Init() has been called */
  t_speed* cells  = NULL;    /* grid containing fluid densities */

  /* Initialize MPI */
  MPI_Init(&argc, &argv);

  /* check whether the initialisation was successful */
  MPI_Initialized(&flag);
  if ( flag != 1 ) {
    MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
  }

  int rank_id;               /* 'rank' of process among it's cohort */ 
  int number_of_ranks;               /* size of cohort, i.e. num processes started */
  MPI_Comm_rank(MPI_COMM_WORLD,&rank_id);
  MPI_Comm_size(MPI_COMM_WORLD,&number_of_ranks);

  initialise(&cells, rank_id);

  MPI_Datatype MPI_T_SPEEDS;
  MPI_Aint displacements[1];
  const int nr_blocks = 1;
  int blocklengths[1] = {NSPEEDS};
  MPI_Datatype oldtypes[1] = {MPI_FLOAT};

  displacements[0] = offsetof(t_speed, speeds);

  int send_rank_id = rank_id - 1;
  if (send_rank_id == -1) {
    send_rank_id = number_of_ranks - 1;
  }
  int rec_rank_id = (rank_id + 1) % number_of_ranks;
  
  MPI_Type_create_struct(nr_blocks, blocklengths, displacements,
                   oldtypes, &MPI_T_SPEEDS);
  MPI_Type_commit(&MPI_T_SPEEDS);
  
  MPI_Status status;
  MPI_Sendrecv(
      // Send the first non halo row
      &cells[0],  // src data
      NX,  // amount of data to send
      MPI_T_SPEEDS,  // data type
      send_rank_id,  // Which rank to send to
      0,
      
      // Recieve and store in halo top row
      &cells[0],
      NX,  // amount of data
      MPI_T_SPEEDS,  // data type
      rec_rank_id,  // Which rank to recieve from 
      0,

      //
      MPI_COMM_WORLD,
      &status
  );


  printf("Value in cell 0, 0 for rank %d: %f\n", rank_id, cells[0].speeds[0]);
  printf("Value in cell 1, 0 for rank %d: %f\n", rank_id, cells[NX].speeds[0]);
  printf("Process %d of %d.\n", rank_id, number_of_ranks);
  
  MPI_Finalize();
}
