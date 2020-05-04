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

#define BILLION 1000000000L

/* USER DEFINED PARAMETERS */
#define N      4 /* Matrix size */
#define ROUNDS 5  /* Number of Rounds */

void srand48(long int seedval);
int rand();
void print_grid(int *A);
void copy_grid(int *old, int *new);
int get_num_live_neighbors(int *A, int r, int c);

int main(){
	struct timespec start, end;
	double diff;
	int state, num_neighbors;

	int r,c;

	/* Dynamically allocate Game of Life Grid*/
	int *A     = (int*)malloc(N * N * sizeof(int));
	int *A_new = (int*)malloc(N * N * sizeof(int));

	if(A == NULL || A_new == NULL){
		printf("Memory not allocated. \n");
		return 0;
	}

	/* Randomly initialize grid */
	srand48(1);
	for(r = 0; r < N; r++){
		for(c = 0; c < N; c++)
			A[r * N + c] = rand() % 2;
	}

	/* Start Timer */
	clock_gettime(CLOCK_MONOTONIC, &start);

	/* Run Game of Life for set number of iterations */
	for(int i = 0; i < ROUNDS; i++){
		printf("ROUND %d\n", i);
		print_grid(A);

		for(r = 0; r < N; r++){
			for(c = 0; c < N; c++){
				printf("hello");
				num_neighbors = get_num_live_neighbors(A, r, c);
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
		copy_grid(A, A_new);
	}

	/* End timer, calculate runtime */
	clock_gettime(CLOCK_MONOTONIC, &end);
	diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
	diff /= BILLION;
	printf("%d rounds of Game of Life", ROUNDS);
	printf("%8f sec", diff);

	free((void *)A);
	free((void *)A_new);

	return 0;  
}

void print_grid(int *A){
	for(int r = 0; r < N; r++){
		for(int c = 0; c < N; c++){
			printf("%d ", A[r * N + c]);
			if(c == N-1){ printf("\n"); }
		}
	}
}

/* Copy new grid into old grid */
void copy_grid(int *old, int *new){
	for(int r = 0; r < N; r++){
		for(int c = 0; c < N; c++)
			old[r * N + c] = new[r * N + c];
	}
}

/* Find number of neighbors that are alive given 
 * a grid and set of coordinates
 */
int get_num_live_neighbors(int *A, int r, int c){
	int num_alive = 0;
	for(int nr = r-1; nr <= r+1; nr++){
		for(int nc = c-1; nr <= c+1; nc++){
			if(nr >= 0 && nr < N && nc >= 0 && nc < N){
				if(nr != r && nc != c){
					if(A[nr * N + nc] == 1)
						num_alive++;
				}
			}
		}
	}
	return num_alive;
}

