#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <assert.h>

typedef struct
{
  float speeds[8];
} t_speed;


typedef struct
{
  float *a;
} t_speed_soa;


int init(t_speed** cells, t_speed** cells2) {
  for (int jj = 0; jj < 2; jj++)
  {
    for (int ii = 0; ii < 2; ii++)
    {
      (*cells)[ii + jj*2].speeds[0] = 1;
      (*cells)[ii + jj*2].speeds[1] = 1;
      (*cells)[ii + jj*2].speeds[2] = 1;
      (*cells)[ii + jj*2].speeds[3] = 1;
      
      (*cells2)[ii + jj*2].speeds[0] = 2;
      (*cells2)[ii + jj*2].speeds[1] = 2;
      (*cells2)[ii + jj*2].speeds[2] = 2;
      (*cells2)[ii + jj*2].speeds[3] = 2;
    }
  }
  return 0;
}

int swap(t_speed** cells, t_speed** cells2) {
  t_speed *tmp = *cells;
  *cells = *cells2;
  *cells2 = tmp;
  return 0;
}

int do_some_thing(t_speed* cells, t_speed* cells2) {
  for (int jj = 0; jj < 2; jj++)
  {
    for (int ii = 0; ii < 2; ii++)
    {
      cells2[ii + jj*2].speeds[0] = 3;
    }
  }
}

int do_some_things(t_speed** cells, t_speed** cells2) {
  do_some_thing(*cells, *cells2);
  swap(cells, cells2);
}


int main_aop(int argc, char* argv[])
{
  printf("hello world");
  t_speed* cells = (t_speed*)malloc(sizeof(t_speed) * 4);
  t_speed* cells2 = (t_speed*)malloc(sizeof(t_speed) * 4);
  printf("hello world");
  

  init(&cells, &cells2);

  assert(cells[0].speeds[0] == 1);
  assert(cells2[0].speeds[0] == 2);

  do_some_things(&cells, &cells2);

  // swap(&cells, &cells2);

  assert(cells2[0].speeds[0] == 1);
  assert(cells[0].speeds[0] == 3);
  
  printf("Assertions complete");
  return 0;
}


void init_soa(t_speed_soa *cells) {
  cells->a = malloc(sizeof(float) * 10);
  cells->a[0] = 0.1f;
  printf("Assertions complete, %.6f\n", cells->a[0]);
}


int main(int argc, char* argv[]) {
  t_speed_soa *cells = malloc(sizeof(t_speed_soa));
  init_soa(cells);
  printf("Assertions complete, %.6f\n", cells->a[0]);
}
