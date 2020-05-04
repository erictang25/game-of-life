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
