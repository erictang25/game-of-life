
// test blinker
#define T_DIM 5
#define T_ROUNDS 3
uint8_t test_1[T_ROUNDS][T_DIM] = { 
  { 0b00000000,
    0b00000000,
    0b00111000,
    0b00000000,
    0b00000000 
  }, 
  { 0b00000000,
    0b00010000,
    0b00010000,
    0b00010000,
    0b00000000
  },
  { 0b00000000,
    0b00000000,
    0b00111000,
    0b00000000,
    0b00000000
  }
}; 

// // test toad
// int test_2[T1_ROUNDS][T1_DIM*T1_DIM] = { 
//   { 0, 0, 0, 0, 0,
//     0, 0, 0, 0, 0,
//     0, 0, 1, 1, 1,
//     0, 1, 1, 1, 0,
//     0, 0, 0, 0, 0 
//   }, 
//   { 0, 0, 0, 0, 0,
//     0, 0, 0, 1, 0,
//     0, 1, 0, 0, 1,
//     0, 1, 0, 0, 1,
//     0, 0, 1, 0, 0 
//   },
//   { 0, 0, 0, 0, 0,
//     0, 0, 0, 0, 0,
//     0, 0, 1, 1, 1,
//     0, 1, 1, 1, 0,
//     0, 0, 0, 0, 0 
//   }
// };

// // test beacon
// int test_3[T1_ROUNDS][T1_DIM*T1_DIM] = { 
//   { 0, 0, 0, 0, 0,
//     0, 1, 1, 0, 0,
//     0, 1, 0, 0, 0,
//     0, 0, 0, 0, 1,
//     0, 0, 0, 1, 1 
//   }, 
//   { 0, 0, 0, 0, 0,
//     0, 1, 1, 0, 0,
//     0, 1, 1, 0, 0,
//     0, 0, 0, 1, 1,
//     0, 0, 0, 1, 1 
//   }, 
//   { 0, 0, 0, 0, 0,
//     0, 1, 1, 0, 0,
//     0, 1, 0, 0, 0,
//     0, 0, 0, 0, 1,
//     0, 0, 0, 1, 1 
//   } 
// }; 

// // test still block
// int test_4[T1_ROUNDS][T1_DIM*T1_DIM] = { 
//   { 0, 0, 0, 0, 0,
//     1, 1, 0, 0, 0,
//     1, 1, 0, 0, 0,
//     0, 0, 0, 1, 1,
//     0, 0, 0, 1, 1 
//   }, 
//   { 0, 0, 0, 0, 0,
//     1, 1, 0, 0, 0,
//     1, 1, 0, 0, 0,
//     0, 0, 0, 1, 1,
//     0, 0, 0, 1, 1 
//   }, 
//   { 0, 0, 0, 0, 0,
//     1, 1, 0, 0, 0,
//     1, 1, 0, 0, 0,
//     0, 0, 0, 1, 1,
//     0, 0, 0, 1, 1 
//   } 
// };

// // test still beehive
// int test_5[T1_ROUNDS][T1_DIM*T1_DIM] = { 
//   { 0, 0, 0, 0, 0,
//     0, 0, 1, 1, 0,
//     0, 1, 0, 0, 1,
//     0, 0, 1, 1, 0,
//     0, 0, 0, 0, 0 
//   }, 
//   { 0, 0, 0, 0, 0,
//     0, 0, 1, 1, 0,
//     0, 1, 0, 0, 1,
//     0, 0, 1, 1, 0,
//     0, 0, 0, 0, 0 
//   }, 
//   { 0, 0, 0, 0, 0,
//     0, 0, 1, 1, 0,
//     0, 1, 0, 0, 1,
//     0, 0, 1, 1, 0,
//     0, 0, 0, 0, 0 
//   } 
// };

// // test still loaf
// int test_7[T1_ROUNDS][T1_DIM*T1_DIM] = { 
//   { 0, 0, 0, 0, 0,
//     0, 1, 1, 0, 0,
//     1, 0, 0, 1, 0,
//     0, 1, 0, 1, 0,
//     0, 0, 1, 0, 0 
//   }, 
//   { 0, 0, 0, 0, 0,
//     0, 1, 1, 0, 0,
//     1, 0, 0, 1, 0,
//     0, 1, 0, 1, 0,
//     0, 0, 1, 0, 0 
//   }, 
//   { 0, 0, 0, 0, 0,
//     0, 1, 1, 0, 0,
//     1, 0, 0, 1, 0,
//     0, 1, 0, 1, 0,
//     0, 0, 1, 0, 0 
//   } 
// };

// // test still boat
// int test_8[T1_ROUNDS][T1_DIM*T1_DIM] = { 
//   { 0, 0, 0, 0, 0,
//     0, 1, 1, 0, 0,
//     0, 1, 0, 1, 0,
//     0, 0, 1, 0, 0,
//     0, 0, 0, 0, 0 
//   }, 
//   { 0, 0, 0, 0, 0,
//     0, 1, 1, 0, 0,
//     0, 1, 0, 1, 0,
//     0, 0, 1, 0, 0,
//     0, 0, 0, 0, 0 
//   }, 
//   { 0, 0, 0, 0, 0,
//     0, 1, 1, 0, 0,
//     0, 1, 0, 1, 0,
//     0, 0, 1, 0, 0,
//     0, 0, 0, 0, 0 
//   } 
// };

// // test still tub
// int test_9[T1_ROUNDS][T1_DIM*T1_DIM] = { 
//   { 0, 0, 0, 0, 0,
//     0, 0, 1, 0, 0,
//     0, 1, 0, 1, 0,
//     0, 0, 1, 0, 0,
//     0, 0, 0, 0, 0 
//   }, 
//   { 0, 0, 0, 0, 0,
//     0, 0, 1, 0, 0,
//     0, 1, 0, 1, 0,
//     0, 0, 1, 0, 0,
//     0, 0, 0, 0, 0 
//   }, 
//   { 0, 0, 0, 0, 0,
//     0, 0, 1, 0, 0,
//     0, 1, 0, 1, 0,
//     0, 0, 1, 0, 0,
//     0, 0, 0, 0, 0 
//   } 
// };