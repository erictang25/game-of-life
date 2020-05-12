/* ECE 5720 Final Project Game of Life Parallel LUT Version
 * Parallel version using Lookup Table
 * 
 * To compile:
 *
 *
 * Authors: Eric Tang (et396), Xiaoyu Yan (xy97)
 * Date:    10 May 2020 
 */

#include <stdint.h>
#include "utils.h"
#include "test_case_bits.h"
#include "lut.h"
   
/* Each byte is 8 cells, each cell is one bit
 * curr_world  : shared array for the entire grid
 * next_world  : next shared array for the entire grid
 * num_bytes   : number of bytes to iterate through for this kernel,
 *               each byte is 8 cells
 * world_length: length of the grid in bytes  
 */
__global__ void gol_lut_cycle(uint8_t *curr_world, uint8_t *next_world, 
                              uint8_t *LUT, uint64_t num_bytes,  
                              uint64_t world_length, uint64_t arr_length ){

  uint64_t grid_idx = threadIdx.x + blockIdx.x * blockDim.x;

  uint64_t start = grid_idx * num_bytes; 
  uint64_t end   = start + num_bytes;
  
  register uint8_t curr_states, state_N, state_S, state_E, state_W, 
                    state_SW, state_SE, state_NW, state_NE; 
  
  uint16_t top, mid, bot;
  uint32_t lut_index;
  for(uint64_t i = start; i < end; i++){
  // TODO Find 3x10 array of cell states
  // TODO Find new byte in LUT
    curr_states = curr_world[i];
    state_N = 0;
    state_S = 0; 
    state_E = 0; 
    state_W = 0; 
    state_SW = 0; 
    state_SE = 0; 
    state_NW = 0; 
    state_NE = 0;
    if ( i > (world_length - 1) ){
      state_N = curr_world[i-world_length];
    }

    if ( i < (arr_length - world_length) ){
      state_S = curr_world[i+world_length];
    }
    // look west
    if ( i % world_length != 0 ){
      state_W  = curr_world[i-1] & 0x1;
      state_NW = curr_world[i-world_length-1] & 0x1;
      state_SW = curr_world[i+world_length-1] & 0x1;
    }
    // Look east
    if ((i + 1) % world_length != 0){
      state_E  = (curr_world[i+1] >> 7) & 0x1;
      state_NE = (curr_world[i-world_length+1] >> 7) & 0x1;
      state_SE = (curr_world[i-world_length+1] >> 7) & 0x1; 
    }
    top = (state_NW << 9) | (state_N << 1) | state_NE;
    mid = (state_W << 9) | (curr_states << 1) | state_E;
    bot = (state_SW << 9) | (state_S << 1) | state_SE;
    lut_index = (top << 20) | (mid << 10) | bot;
    next_world[i] = LUT[lut_index];
  }
}

/* Run Game of Life Simulation 
 * return 1 if error occurs
 */
int gol_lut( uint8_t *world, uint64_t N, uint64_t P, int rounds, int test, 
             uint8_t **ref, int trace ){

	printf("Begin running LUT\n");

	struct timespec t_start, t_end;  
	long double average, start, end;

	/* Set grid parameters */
	uint64_t world_length = N/8;
  uint64_t num_elements = N*N/8;
  int blocks = 1;
  while ( P > 1024 ){
    P -= 1024;
    blocks ++;
	}

	/* LUT parameters */
	uint64_t LUT_size = pow(2,30);
	uint64_t num_entries = LUT_size / P;

  dim3 Block(blocks); // Square pattern
  dim3 Grid(P);

  /* Dynamically allocate memory for world */
  uint8_t *dev_curr_world, *dev_next_world, *tmp, *LUT;
  cudaMalloc((void **) &dev_curr_world, num_elements*sizeof(uint8_t)); 
	cudaMalloc((void **) &dev_next_world, num_elements*sizeof(uint8_t)); 
	cudaMalloc((void **) &tmp, num_elements*sizeof(uint8_t)); 
	cudaMalloc((void **) &LUT, LUT_size*sizeof(uint8_t)); 

  cudaMemcpy(dev_curr_world, world, num_elements*sizeof(uint8_t), cudaMemcpyHostToDevice);

	/* Precompute LUT */
	printf("Calculating LUT\n");
	compute10x3LUTKernel<<<Grid, Block>>>(LUT, num_entries);
	printf("Finished calculating LUT\n");

  clock_gettime(CLOCK_MONOTONIC, &t_start); /* Start timer */
	for ( int i = 0; i < rounds; i++ ){
    	gol_lut_cycle<<<Grid, Block>>>(dev_curr_world, dev_next_world, LUT,
                                       N*N/8/P, world_length, num_elements);
    	cudaMemcpy(world, dev_next_world, num_elements*sizeof(uint8_t), cudaMemcpyDeviceToHost);
    	tmp = dev_curr_world;
    	dev_curr_world = dev_next_world;
    	dev_next_world = tmp;
    	if (test && !world_bits_correct(world, ref[i], N)) 
			return 1;  /* return 1 if error occurs */
    	if (trace) 
	    	print_world_bits(world, N);
	}
  clock_gettime(CLOCK_MONOTONIC, &t_end); /* End timer */

	/* Calculate average runtime per round */
	start = t_start.tv_sec + (long double)t_start.tv_nsec/BILLION;
	end = t_end.tv_sec + (long double)t_end.tv_nsec/BILLION;
	average = (end - start)/((long double)rounds);
	  
  // printf("Grid Size: %ldx%ld, # Rounds: %d, # Threads: %ld\n", N, N, rounds, P*blocks);
  // printf("Average time per round: %.13LFs\n", average);
  printf("4|%d|%ld|%ld|%.13LF|\n", rounds, N, P*blocks, average );
  return 0;
}

int main( int argc, char** argv ){
	// Default values
  	int test   = 0; // Run direct test cases
  	uint64_t N = 8; // Matrix size 
  	uint64_t P = 1; // number of threads
 	int ROUNDS = 5; // Number of Rounds
  	int trace  = 0; // print trace of world  

	/* Set Game of Life parameters acording to user input */
  if ( argc > 1 ) test = atoi(argv[1]); 
  if ( argc > 2 ) {
    N = atoi(argv[2]); // Dimensions of the block
    if ( N % 8 != 0 ){
      printf( "Invalid N:[%ld]; must be divisible by 8\n", N );
      N = 8;
    }	 
  }
  if ( argc > 3 ) ROUNDS = atoi(argv[3]); 
	if ( argc > 4 ) { 
    P = atoi(argv[4]); // number of threads
    if ( P > N*N/8 ){
        printf( "Invalid P:[%ld]; Too many threads for number of elements %ld\n", P, N*N/8 );
        return 1;
    }
    if ( N*N/8 % P != 0 ){
        printf( "Invalid P:[%ld]; Number of threads should be a factor of %ld\n", P, N*N/8 );
        return 1;
    }
  }
  if ( argc > 5 ) trace  = atoi(argv[5]); 

  uint8_t *world, **ref;

  	if (test){
  	/* Setup and run all test cases*/
    	int num_correct = 0;
    	N      = T_DIM;
    	ROUNDS = T_ROUNDS - 1;

    	// Test 1
    	printf("Running test 1\n");
    	world  = test_1[0];
    	ref    = (uint8_t**) malloc( sizeof(uint8_t*) * ROUNDS );
    	for (int r = 0; r < ROUNDS; r++)
        	ref[r] = test_1[r+1];
    	if (!gol_lut(world, N, P, ROUNDS, test, ref, trace)) 
    		num_correct++;
	}

//    // Test 2
//    printf("Running test 2\n");
//    world  = test_2[0];
//    for ( int r = 0; r < ROUNDS; r++ )
//        ref[r] = test_2[r+1];
//    if (!gol_lut( world, N, P, ROUNDS, test, ref, trace )) 
//    num_correct++;
//
//    // Test 3
//    printf("Running test 3\n");
//    world  = test_3[0];
//    for ( int r = 0; r < ROUNDS; r++ )
//        ref[r] = test_3[r+1];
//    if (!gol_lut( world, N, P, ROUNDS, test, ref, trace )) 
//    num_correct++;
//
//    // Test 4
//    printf("Running test 4\n");
//    world  = test_4[0];
//    for ( int r = 0; r < ROUNDS; r++ )
//        ref[r] = test_4[r+1];
//    if (!gol_lut( world, N, P, ROUNDS, test, ref, trace )) 
//    num_correct++;
//
//    // Test 5
//    printf("Running test 5\n");
//    world  = test_5[0];
//    for ( int r = 0; r < ROUNDS; r++ )
//        ref[r] = test_5[r+1];
//    if (!gol_lut( world, N, P, ROUNDS, test, ref, trace )) 
//    num_correct++;
//
//    // Test 6
//    printf("Running test 6\n");
//    world  = test_6[0];
//    for ( int r = 0; r < ROUNDS; r++ )
//        ref[r] = test_6[r+1];
//    if (!gol_lut( world, N, P, ROUNDS, test, ref, trace )) 
//    num_correct++;
//
//    // Test 7
//    printf("Running test 7\n");
//    world  = test_7[0];
//    for ( int r = 0; r < ROUNDS; r++ )
//        ref[r] = test_7[r+1];
//    if (!gol_lut( world, N, P, ROUNDS, test, ref, trace )) 
//    num_correct++;
//
//    // Test 8
//    printf("Running test 8\n");
//    world  = test_8[0];
//    for ( int r = 0; r < ROUNDS; r++ )
//        ref[r] = test_8[r+1];
//    if (!gol_lut( world, N, P, ROUNDS, test, ref, trace )) 
//    num_correct++;
//
//    // Test 9
//    printf("Running test 9\n");
//    world  = test_9[0];
//    N      = T9_DIM;
//    ROUNDS = T9_ROUNDS - 1;
//    for ( int r = 0; r < ROUNDS; r++ )
//      ref[r] = test_9[r+1];
//    if (!gol_lut( world, N, P, ROUNDS, test, ref, trace)) 
//      num_correct++;
//  
//    printf("%s %d/9 tests passed %s\n", KBLU, num_correct, KNRM);
//    free(ref);
//  } else {
//    /* Run simulation w/ random seed */
//    uint64_t n_elements = N*N/8;
//    
//    /* Initialize world with random states*/
//    srand48(1);
//    world = (uint8_t*)malloc(n_elements * sizeof(uint8_t));
//    for(int r = 0; r < N; r++){
//      for(int c = 0; c < N/8; c++)
//        world[r*N/8+c] = (uint8_t)(rand() % 256);
//    } 
//    ref = NULL;
//    if (trace) 
//      print_world_bits( world, N ); /* Print initial state*/
//    gol_lut( world, N, P, ROUNDS, test, ref, trace ); /* Simulate Game of Life*/
//    free(world);
//  }

  return 0;
}
