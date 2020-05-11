/* ECE 5720 Final Project Game of Life Parallel LUT Version
 * Parallel version using Lookup Table
 * 
 * Authors: Eric Tang (et396), Xiaoyu Yan (xy97)
 * Date:    10 May 2020 
 */

#include <stdint.h>
#include "utils.h"
#include "test_case_bits.h"


/* Each byte is 8 cells, each cell is one bit
 * curr_world  : shared array for the entire grid
 * next_world  : next shared array for the entire grid
 * num_bytes   : number of bytes to iterate through for this kernel,
 *               each byte is 8 cells
 * world_length: length of the ????  
 */
__global__ void gol_lut_cycle( uint8_t *curr_world, uint8_t *next_world, uint64_t num_bytes, 
                           uint64_t world_length, uint64_t arr_length ){
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  // int offset = blockDim.x * gridDim.x;
  register uint8_t curr_8_states, curr_bit = 0, next_8_states = 0,
                   num_alive = 0, north_byte, south_byte, threshold = 4;
  uint8_t NE_byte, SE_byte, NW_byte, SW_byte;
  uint64_t start = x*num_bytes; 
  uint64_t end   = start + num_bytes;   
  // printf("nb[%d] s[%d] e[%d] x:[%d] \n", num_bytes, start, end, x);                       
  for ( uint64_t i = start; i < end; i++ ){
    curr_8_states = curr_world[i];
    next_8_states = curr_8_states;
    for (int bit = 0; bit < 8; bit ++){
      // iterate through each bit
      curr_bit = (curr_8_states >> bit) & 0x1; // extract cell state 
      num_alive = 0;
      // Check west and east first since they are in register
      // look west in same register
      if ( bit != 7 )
        if ( (curr_8_states >> (bit+1)) & 0x1 ) num_alive++;
      
      // look east in same register
      if ( bit != 0 )
        if ( (curr_8_states >> (bit-1)) & 0x1 ) num_alive++;
        
      // look north
      if ( i > (world_length - 1) ){
        north_byte = curr_world[i-world_length];
        if ( (north_byte >> (bit))   & 0x1 ) num_alive++;
        if ( bit == 0 ){
          if ( (north_byte >> (bit+1)) & 0x1 ) num_alive++;
        }
        else if ( bit == 7 ){
          if ( (north_byte >> (bit-1)) & 0x1 ) num_alive++;
        }
        else{
          if ( (north_byte >> (bit+1)) & 0x1 ) num_alive++;
          if ( (north_byte >> (bit-1)) & 0x1 ) num_alive++;
        }
      } 

      // look south but don't have to if already past threshold
      if ( (i < (arr_length - world_length)) && (num_alive < threshold) ){
        south_byte = curr_world[i+world_length];
        if ( (south_byte >> (bit))   & 0x1 ) num_alive++;
        if ( bit == 0 ){
          if ( (south_byte >> (bit+1)) & 0x1 ) num_alive++;
        }
        else if ( bit == 7 ){
          if ( (south_byte >> (bit-1)) & 0x1 ) num_alive++;
        }
        else{
          if ( (south_byte >> (bit-1)) & 0x1 ) num_alive++;
          if ( (south_byte >> (bit+1)) & 0x1 ) num_alive++;
        }
      }

      // look west in diff reg
      if ( (bit == 7) && (i % world_length != 0) && (num_alive < threshold) ){
        // for the lsb
        if ( curr_world[i-1] & 0x1 ) num_alive++;
        NW_byte = curr_world[i-world_length-1];
        if ( NW_byte & 0x1 ) num_alive++;
        SW_byte = curr_world[i+world_length-1];
        if ( SW_byte & 0x1 ) num_alive++;
      }
      
      // look east in diff reg
      if ( (bit == 0) && ((i + 1) % world_length != 0) && (num_alive < threshold) ){
        // for the msb
        if ( (curr_world[i-1] >> 7) & 0x1 ) num_alive++;
        NE_byte = curr_world[i-world_length+1];
        if ( (NE_byte >> 7) & 0x1 ) num_alive++;
        SE_byte = curr_world[i+world_length+1];
        if ( (SE_byte >> 7) & 0x1 ) num_alive++;
      }

      // Determine if an alive cell will die 
      if (curr_bit && (num_alive < 2 || num_alive > 3)) 
        next_8_states ^= (uint8_t)(1 << bit);

      // Determine if a dead cell will be alive
      else if ( !curr_bit && (num_alive == 3))
        next_8_states ^= (uint8_t)(1 << bit);
    }
    next_world[i] = next_8_states;
  }
}

/* Run Game of Life Simulation 
 * return 1 if error occurs
 */
int gol_lut( uint8_t *world, uint64_t N, uint64_t P, int rounds, int test, 
             uint8_t **ref, int trace ){

	/* Set grid parameters */
	uint64_t world_length = N/8;
  	uint64_t num_elements = N*N/8;
  	struct timespec t_start, t_end;
	long double average = 0, st, ed, diff;
  	int blocks = P / 1024;

  	dim3 Block(blocks); // Square pattern
  	dim3 Grid(P);

  	uint8_t *dev_curr_world, *dev_next_world;
  	cudaMalloc((void **) &dev_curr_world, num_elements*sizeof(uint8_t)); 
  	cudaMalloc((void **) &dev_next_world, num_elements*sizeof(uint8_t)); 
  
	for ( int i = 0; i < rounds; i++ ){
    	clock_gettime(CLOCK_MONOTONIC, &t_start); /* Start Timer */
    	cudaMemcpy(dev_curr_world, world, num_elements*sizeof(uint8_t), cudaMemcpyHostToDevice);
    	gol_lut_cycle<<<Grid, Block>>>(dev_curr_world, dev_next_world, N*N/8/P, 
                                       world_length, num_elements);
    	cudaMemcpy(world, dev_next_world, num_elements*sizeof(uint8_t), cudaMemcpyDeviceToHost);
    	clock_gettime(CLOCK_MONOTONIC, &t_end);   /* End timer */
    	st = t_start.tv_sec + (long double)t_start.tv_nsec/BILLION;
    	ed = t_end.tv_sec + (long double)t_end.tv_nsec/BILLION;
    	diff = ed - st;
    	average += diff/((long double)rounds);
    	if (test && !world_bits_correct(world, ref[i], N)) 
			return 1;
    	if (trace) 
			print_world_bits(world, N);
  	}
  	printf("Grid Size: %ldx%ld, # Rounds: %d, # Threads: %ld\n", N, N, rounds, P*blocks);
  	printf("Average time per round: %.13LFs\n", average);
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
	if ( argc > 3 ) { 
    	P = atoi(argv[3]); // number of threads
    	if ( P > N*N/8 ){
      		printf( "Invalid P:[%ld]; Too many threads for number of elements %ld\n", P, N*N/8 );
      		return 1;
    	}
    	if ( N*N/8 % P != 0 ){
      		printf( "Invalid P:[%ld]; Number of threads should be a factor of %ld\n", P, N*N/8 );
      		return 1;
    	}
  	}
  	if ( argc > 4 ) ROUNDS = atoi(argv[4]); 
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
    	for ( int r = 0; r < ROUNDS; r++ )
      		ref[r] = test_1[r+1];
    	if (!gol_lut( world, N, P, ROUNDS, test, ref, trace )) 
			num_correct++;

    	// Test 2
    	printf("Running test 2\n");
    	world  = test_2[0];
    	for ( int r = 0; r < ROUNDS; r++ )
      		ref[r] = test_2[r+1];
    	if (!gol_lut( world, N, P, ROUNDS, test, ref, trace )) 
			num_correct++;

    	// Test 3
    	printf("Running test 3\n");
    	world  = test_3[0];
    	for ( int r = 0; r < ROUNDS; r++ )
      		ref[r] = test_3[r+1];
    	if (!gol_lut( world, N, P, ROUNDS, test, ref, trace )) 
			num_correct++;

    	// Test 4
    	printf("Running test 4\n");
    	world  = test_4[0];
    	for ( int r = 0; r < ROUNDS; r++ )
      		ref[r] = test_4[r+1];
    	if (!gol_lut( world, N, P, ROUNDS, test, ref, trace )) 
			num_correct++;

    	// Test 5
    	printf("Running test 5\n");
    	world  = test_5[0];
    	for ( int r = 0; r < ROUNDS; r++ )
      		ref[r] = test_5[r+1];
    	if (!gol_lut( world, N, P, ROUNDS, test, ref, trace )) 
			num_correct++;

		// Test 6
    	printf("Running test 6\n");
    	world  = test_6[0];
    	for ( int r = 0; r < ROUNDS; r++ )
      		ref[r] = test_6[r+1];
    	if (!gol_lut( world, N, P, ROUNDS, test, ref, trace )) 
			num_correct++;

    	// Test 7
    	printf("Running test 7\n");
    	world  = test_7[0];
    	for ( int r = 0; r < ROUNDS; r++ )
      		ref[r] = test_7[r+1];
    	if (!gol_lut( world, N, P, ROUNDS, test, ref, trace )) 
			num_correct++;

    	// Test 8
    	printf("Running test 8\n");
    	world  = test_8[0];
    	for ( int r = 0; r < ROUNDS; r++ )
      		ref[r] = test_8[r+1];
    	if (!gol_lut( world, N, P, ROUNDS, test, ref, trace )) 
			num_correct++;

    	// Test 9
    	printf("Running test 9\n");
    world  = test_9[0];
    N      = T9_DIM;
    ROUNDS = T9_ROUNDS - 1;
    for ( int r = 0; r < ROUNDS; r++ )
      ref[r] = test_9[r+1];
    if (!gol_lut( world, N, P, ROUNDS, test, ref, trace )) num_correct++;
    
    printf("%s %d/9 tests passed %s\n", KBLU, num_correct, 9, KNRM);
    free(ref);

  } else {
	/* Run simulation w/ random seed */
    srand48(1);
    uint64_t n_elements = N*N/8;
    world = (uint8_t*)malloc(n_elements * sizeof(uint8_t));
    for(int r = 0; r < N; r++){
      for(int c = 0; c < N/8; c++)
        world[r*N/8+c] = (uint8_t)(rand() % 256);
    } 
    ref   = NULL;
    if (trace) print_world_bits( world, N );
    gol_lut( world, N, P, ROUNDS, test, ref, trace );
    free(world);
  }
  return 0;
}

