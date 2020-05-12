/*
 * Eric Tang (et396), Xiaoyu Yan (xy97) 
 * ECE 5720 Final Project Game of Life Sequential Version
 * Sequential version using bit per cell implementation
 * 
 * To compile:
 * 
 * 
 */

#include <time.h>
#include <stdint.h>
#include "utils.h"
#include "test_case_bits.h"
#include <string.h>

void gol_cycle_seq( uint8_t *curr_world, uint8_t *next_world, uint64_t num_bytes, 
                    uint64_t world_length, uint64_t arr_length ){
  register uint8_t curr_8_states, curr_bit = 0, next_8_states = 0,
                   num_alive = 0, north_byte, south_byte, threshold = 4;
  uint8_t NE_byte, SE_byte, NW_byte, SW_byte;
  int start = 0; 
  int end   = start + num_bytes;                         
  for ( int i = start; i < end; i++ ){
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
        if ( (curr_world[i+1] >> 7) & 0x1 ) num_alive++;
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

int gol_bit_per_cell_seq( uint8_t *world, uint64_t N, int rounds, int test, 
                      uint8_t **ref, int trace ){
  uint64_t world_length = N/8;
  uint64_t num_elements = N*N/8;
  struct timespec t_start, t_end;
	long double average = 0, st, ed, diff;
  uint8_t *dev_next_world = (uint8_t*)malloc(num_elements * sizeof(uint8_t));
  for ( int i = 0; i < rounds; i++ ){
    clock_gettime(CLOCK_MONOTONIC, &t_start); /* Start Timer */
    gol_cycle_seq( world, dev_next_world, num_elements, world_length, num_elements );
    memcpy(world, dev_next_world, num_elements*sizeof(uint8_t) );
    clock_gettime(CLOCK_MONOTONIC, &t_end);   /* End timer */
    st = t_start.tv_sec + (long double)t_start.tv_nsec/BILLION;
    ed = t_end.tv_sec + (long double)t_end.tv_nsec/BILLION;
    diff = ed - st;
    average += diff/((long double)rounds);
    // print_world_bits(ref, N);
    if ( test && !world_bits_correct(world, ref[i], N)) return 1;
    if ( trace ) print_world_bits(world, N);
  }
  printf("\nGrid Size: %ldx%ld, # Rounds: %d\n", N, N, rounds);
  printf("Average time per round: %.13LFs\n", average);
  free(dev_next_world);
  return 0;
}

int main( int argc, char** argv ){
  // Default values
  int test   = 0; // Run direct test cases
  uint64_t N = 8; // Matrix size 
  int ROUNDS = 5; // Number of Rounds
  int trace  = 0; // print trace of world  
  if ( argc > 1 ) test = atoi(argv[1]); 
  if ( argc > 2 ) {
    N = atoi(argv[2]); // Dimensions of the block
    if ( N % 8 != 0 ){
      printf( "Invalid N:[%ld]; must be divisible by 8\n", N );
      N = 8;
    } 
  }
  if ( argc > 3 ) ROUNDS = atoi(argv[3]); 
  if ( argc > 4 ) trace  = atoi(argv[4]); 
  uint8_t *world, **ref;
  if (test){
    int num_correct = 0, num_tests = 0;
    // Test 1
    num_tests++;
    printf("Running test %d\n", num_tests);
    world  = test_1[0];
    N      = T_DIM;
    ROUNDS = T_ROUNDS - 1;
    ref    = (uint8_t**) malloc( sizeof(uint8_t*) * ROUNDS );
    for ( int r = 0; r < ROUNDS; r++ )
      ref[r] = test_1[r+1];
    if (!gol_bit_per_cell_seq( world, N, ROUNDS, test, ref, trace )) num_correct++;
    // Test 2
    num_tests++;
    printf("Running test %d\n", num_tests);
    world  = test_2[0];
    for ( int r = 0; r < ROUNDS; r++ )
      ref[r] = test_2[r+1];
    if (!gol_bit_per_cell_seq( world, N, ROUNDS, test, ref, trace )) num_correct++;
    // T3
    num_tests++;
    printf("Running test %d\n", num_tests);
    world  = test_3[0];
    for ( int r = 0; r < ROUNDS; r++ )
      ref[r] = test_3[r+1];
    if (!gol_bit_per_cell_seq( world, N, ROUNDS, test, ref, trace )) num_correct++;
    // T4
    num_tests++;
    printf("Running test %d\n", num_tests);
    world  = test_4[0];
    for ( int r = 0; r < ROUNDS; r++ )
      ref[r] = test_4[r+1];
    if (!gol_bit_per_cell_seq( world, N, ROUNDS, test, ref, trace )) num_correct++;
    // T5
    num_tests++;
    printf("Running test %d\n", num_tests);
    world  = test_5[0];
    for ( int r = 0; r < ROUNDS; r++ )
      ref[r] = test_5[r+1];
    if (!gol_bit_per_cell_seq( world, N, ROUNDS, test, ref, trace )) num_correct++;
    // T6
    num_tests++;
    printf("Running test %d\n", num_tests);
    world  = test_6[0];
    for ( int r = 0; r < ROUNDS; r++ )
      ref[r] = test_6[r+1];
    if (!gol_bit_per_cell_seq( world, N, ROUNDS, test, ref, trace )) num_correct++;
    // T7
    num_tests++;
    printf("Running test %d\n", num_tests);
    world  = test_7[0];
    for ( int r = 0; r < ROUNDS; r++ )
      ref[r] = test_7[r+1];
    if (!gol_bit_per_cell_seq( world, N, ROUNDS, test, ref, trace )) num_correct++;
    // T8
    num_tests++;
    printf("Running test %d\n", num_tests);
    world  = test_8[0];
    for ( int r = 0; r < ROUNDS; r++ )
      ref[r] = test_8[r+1];
    if (!gol_bit_per_cell_seq( world, N, ROUNDS, test, ref, trace )) num_correct++;
    // T9
    num_tests++;
    printf("Running test %d\n", num_tests);
    world  = test_9[0];
    N      = T9_DIM;
    ROUNDS = T9_ROUNDS - 1;
    for ( int r = 0; r < ROUNDS; r++ )
      ref[r] = test_9[r+1];
    if (!gol_bit_per_cell_seq( world, N, ROUNDS, test, ref, trace )) num_correct++;
    
    printf("%s %d/%d tests passed %s\n", KBLU, num_correct, num_tests, KNRM);
    free(ref);
  }
  else {
    srand48(1);
    int num_elements = N*N/8;
    world = (uint8_t*)malloc(num_elements * sizeof(uint8_t));
    for(int r = 0; r < N; r++){
      for(int c = 0; c < N/8; c++)
        world[r*N/8+c] = (uint8_t)(rand() % 256);
    } 
    ref = NULL;
    if (trace) print_world_bits( world, N );
    gol_bit_per_cell_seq( world, N, ROUNDS, test, ref, trace );
    free(world);
  }
}
