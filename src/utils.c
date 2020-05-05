#include "utils.h"

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

/**
 * prints the world assuming each cell is 1 bit
*/
void print_world_bits( uint8_t *world, int N ){
  int cell;
  // by row
  printf("[\n");
  for ( int i = 0; i < N; i++ ){
    // by col
    for ( int j = 0; j < N/8; j++){
      // by bits
      for ( int bit = 7; bit >= 0; bit--){
        cell = (world[i*N/8+j] >> bit) & 0x1;
        if (cell)
          printf( "%d ", cell );
        else
          printf( "  " );
      }
    }
    printf("\n");
  }
  printf("]\n");
}

int world_bits_correct( uint8_t *test, uint8_t *ref, int N ){
  int match = 1;
  for ( int i = 0; i < N; i++ ){
    for ( int j = 0; j < N/8; j++ ){
      if ( test[i*N/8+j] != ref[i*N/8+j] ){
        match = 0;
        break;
      }
    }
  }
  if (!match) {
    printf("ERROR:\n");
    printf("test:\n");
    print_world_bits(test, N);
    printf("ref:\n");
    print_world_bits(ref, N);
  }
  else{
    printf("%sCORRECT%s\n", KYEL, KNRM);
    print_world_bits(test, N);
  } 
  return match; 
}
