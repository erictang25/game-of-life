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
#include "utils.h"
#include "test_cases.h"


void srand48(long int seedval);
int rand();
void print_grid(int *A, int N);
void copy_grid(int *old, int *new, int N);
int get_num_live_neighbors(int *A, int r, int c, int N);
int game_of_life(int *A, int N, int ROUNDS, int test);

int main( int argc, char** argv ){
	int N = 10;     /* Matrix size */
	int ROUNDS = 5; /* Number of Rounds */
	int test = 0;

	if ( argc > 1 ) test   = atoi(argv[1]); 
	if ( argc > 2 ) N      = atoi(argv[2]); 
	if ( argc > 3 ) ROUNDS = atoi(argv[3]); 
  
	struct timespec start, end;
	double diff;
	int *A;

	/* Initialize Game of Life Grid */
	if (test>0){
		N = T1_DIM;
		ROUNDS = T1_ROUNDS;
		switch(test){
			case 1: A = test_1[0];
				break;
			case 2: A = test_2[0];
				break;
			case 3:
				N = T3_DIM; ROUNDS = T3_ROUNDS; 
				A = test_3[0];
				break;
			case 4:
				N = T4_DIM; ROUNDS = T4_ROUNDS; 
 				A = test_4[0];
				break;
			case 5: A = test_5[0];
				break;
			case 6: 
				N = T6_DIM; ROUNDS = T6_ROUNDS; 
				A = test_6[0];
				break;
			case 7: A = test_7[0];
				break;
			case 8: A = test_8[0];
				break;
			default: printf("Invalid test input");
				return 0;
		}
	} else{
		/* Dynamically allocate Game of Life Grid*/
		A = (int*)malloc(N * N * sizeof(int));
		/* Randomly initialize grid */
    	srand48(1);
    	for(int r = 0; r < N; r++){
      		for(int c = 0; c < N; c++)
        		A[r * N + c] = rand() % 2;
  		}
  		if(A == NULL){
    		printf("Memory not allocated. \n");
    		return 0;
		}
	}

	/* Run and time Game of Life */
	clock_gettime(CLOCK_MONOTONIC, &start); /* Start Timer */
	int status = game_of_life(A, N, ROUNDS, test);
	clock_gettime(CLOCK_MONOTONIC, &end);   /* End timer */

	if(status == -1)
		return 0;

	if(test > 0)
    	printf("%sPASSED TEST %d%s\n", KGRN, test, KNRM);
	else
		free((void *)A);

	/* Calculate runtime */
	diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
	diff /= BILLION;
	printf("%d rounds of Game of Life\n", ROUNDS);
	printf("Runtime: %8f sec\n", diff);
	
	return 0;  
}


int game_of_life(int *A, int N, int ROUNDS, int test){

	int num_neighbors, state;

	int *A_new = (int*)malloc(N * N * sizeof(int));
	if(A_new == NULL){
		printf("Memory not allocated. \n");
		return 0;
	}

	/* Run Game of Life for set number of iterations */
	for(int i = 0; i < ROUNDS; i++){
		printf("ROUND %d\n", i);
		print_grid(A, N);
		printf("\n");

		switch(test){
    		case 1:  
      			if (check_generation_output( A, test_1[i], N ))
        			return -1;
				break;
			case 2:  
      			if (check_generation_output( A, test_2[i], N ))
        			return -1;
				break;
			case 3:  
      			if (check_generation_output( A, test_3[i%2], N ))
        			return -1;
				break;
			case 4:  
      			if (check_generation_output( A, test_4[0], N ))
        			return -1;
				break;
			case 5:  
      			if (check_generation_output( A, test_5[i], N ))
        			return -1;
				break;
			case 6:  
      			if (check_generation_output( A, test_6[0], N ))
        			return -1;
				break;
			case 7:  
   		  	 	if (check_generation_output( A, test_7[i], N ))
        			return -1;
				break;
			case 8:  
    		  	if (check_generation_output( A, test_8[i], N ))
        			return -1;
				break;
		}

		for(int r = 0; r < N; r++){
			for(int c = 0; c < N; c++){
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
	}

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
