#include "utils.h"

/* Check if current world state matches reference, 
 * return 1 if there is an error 
 */
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

/* Print the world assuming each cell is 1 int */
void print_world(int *world, int N){
	int cell;
	for (int r = 0; r < N; r++){
		printf(" ");
    	for(int c = 0; c < N; c++){
        	cell = world[r * N + c];
        	if (cell)
				printf("%s  %s", BGRN, BNRM);
        	else
				printf("%s  %s", BRED, BNRM);
      	}
    	printf("\n");
    }
    printf("\n");
}

/* Print the world assuming each cell is 1 bit */
void print_world_bits( uint8_t *world, int N ){
  int cell;
	for ( int i = 0; i < N; i++ ){
		printf(" ");
    for ( int j = 0; j < N/8; j++){
      for ( int bit = 7; bit >= 0; bit--){
        cell = (world[i*N/8+j] >> bit) & 0x1;
        if (cell)
			printf("%s  %s", BGRN, BNRM);
          // printf( "%sO%s",KYEL,KNRM );
//          printf( "%sO%s",KYEL,KNRM );
        else
//        	printf( " " );
			printf("%s  %s", BRED, BNRM);
      }
    }
    printf("\n");
  }
    printf("\n");
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
  return match; 
}
