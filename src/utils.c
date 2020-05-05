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
      for ( int bit = 0; bit < 8; bit++){
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
