/**
 * Eric Tang (et396), Xiaoyu Yan (xy97) 
 * ECE 5720 Final Project Game of Life Sequential Version
 * 
 * To compile:
 * 
 * 
 * */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test_case1.h"

#define BILLION 1000000000L
#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

void srand48(long int seedval);
int rand();
void print_grid(int *A, int N);
void copy_grid(int *old, int *new, int N);
int get_num_live_neighbors(int *A, int r, int c, int N);

int check_generation_output( int *A, int *ref, int N ){
  int mismatch = 0;
  for ( int i = 0; i < N; i++ ){
    for ( int j = 0; j < N; j++ ){
      if ( A[i*N+j] != ref[i*N+j] )
        mismatch = 1;
    }
  }
  if (mismatch){
    printf("\nERROR\n");
    for (int i = 0; i < N; i++){
      for (int j = 0; j < N; j++){
        if (A[i*N+j] != ref[i*N+j]) 
          printf("%s%d%s ", KRED, A[i*N+j], KNRM);
        else
          printf("%d ", A[i*N+j]);
      }
      printf("\n");
    }
  }
  return mismatch;
}

int main( int argc, char** argv ){
  int N = 10;     /* Matrix size */
  int ROUNDS = 5; /* Number of Rounds */
  int test = 0;
  if ( argc > 1 ) N      = atoi(argv[1]); 
  if ( argc > 2 ) ROUNDS = atoi(argv[2]); 
  if ( argc > 3 ) test   = atoi(argv[3]); 
  
	struct timespec start, end;
	double diff;
	int state, num_neighbors;

	int r,c;
  int *A;
  if (test>0){
    if (test == 1){
      N = T1_DIM;
      ROUNDS = T1_ROUNDS;
      A = test_1[0];
    }
  } else{
    /* Dynamically allocate Game of Life Grid*/
    A = (int*)malloc(N * N * sizeof(int));
		/* Randomly initialize grid */
    srand48(1);
    for(r = 0; r < N; r++){
      for(c = 0; c < N; c++)
        A[r * N + c] = rand() % 2;
    	}
  	}
  	int *A_new = (int*)malloc(N * N * sizeof(int));
  	if(A == NULL || A_new == NULL){
    	printf("Memory not allocated. \n");
    	return 0;
  }

	/* Start Timer */
	clock_gettime(CLOCK_MONOTONIC, &start);

	/* Run Game of Life for set number of iterations */
	for(int i = 0; i < ROUNDS; i++){
		printf("ROUND %d\n", i);
		print_grid(A, N);
		printf("\n");

		for(r = 0; r < N; r++){
			for(c = 0; c < N; c++){
				num_neighbors = get_num_live_neighbors(A, r, c, N);
				state = A[r * N + c];
				if(state == 1 && (num_neighbors == 2 || num_neighbors == 3)){
					A_new[r * N + c] = 1;
				} else if(state == 0 && num_neighbors == 3){
					A_new[r * N + c] = 1;
				} else {
					A_new[r * N + c] = 0;
				}
			}
		}
		copy_grid(A, A_new, N);
    if( test>0 ){ 
      if (check_generation_output( A_new, test_1[i], N ))
        return -1;
    }
	}

	/* End timer, calculate runtime */
	clock_gettime(CLOCK_MONOTONIC, &end);
	diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
	diff /= BILLION;
	printf("%d rounds of Game of Life\n", ROUNDS);
	printf("%8f sec\n", diff);

	free((void *)A);
	free((void *)A_new);

	return 0;  
}

void print_grid( int *A, int N ){
	for(int r = 0; r < N; r++){
		for(int c = 0; c < N; c++){
			printf("%d ", A[r * N + c]);
			if(c == N-1){ printf("\n"); }
		}
	}
}

/* Copy new grid into old grid */
void copy_grid(int *old, int *new, int N){
	for(int r = 0; r < N; r++){
		for(int c = 0; c < N; c++)
			old[r * N + c] = new[r * N + c];
	}
}

/* Find number of neighbors that are alive given 
 * a grid and set of coordinates
 */
int get_num_live_neighbors(int *A, int r, int c, int N){
	int row, col;
	int num_alive = 0;
	for(int nr = r-1; nr <= r+1; nr++){
		for(int nc = c-1; nc <= c+1; nc++){
			row = nr;
			col = nc;

			if(row < 0) 
				row = N-1; 
			else if(row >= N)
				row = 0; 

			if(col < 0)
				col = N-1;
			else if(col >= N)
				col = 0;

			if((row != r || col != c) && A[row * N + col] == 1)
					num_alive++;
		}
	}

	return num_alive;
}
