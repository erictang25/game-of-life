# Game of Life

The Game of Life, also known simply as Life, is a cellular automaton devised by the British mathematician John Horton Conway in 1970. It is a zero-player game, meaning that its evolution is determined by its initial state, requiring no further input. One interacts with the Game of Life by creating an initial configuration and observing how it evolves. It is Turing complete and can simulate a universal constructor or any other Turing machine.

## Getting Started

To run the baseline sequential version

```
cd build
make utils
make gol_seq
make seq_tests
./gol_seq G R
```
where R is the number of rounds and G is the size of one side of the NxN grid. The final command will simulate the baseline sequential version of the Game of Life for the given number of rounds with the corresponding grid size and output the runtime. In order to view the grid at each round, run 
```
./gol_seq G R 1
```

## Developers

Eric Tang (et396)  
Xiaoyu Yan (xy97)
