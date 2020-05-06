
import os

dims      = [ 2**i for i in range(3, 17) ]
nthreads  = [1, 2, 4, 8, 16, 32, 64, 256, 1024, 2048, 4096]

rounds = 10

for d in dims:
  os.system("echo running ./gol_seq 0 {} {} >> gol_seq_results.out".format(d))
  # os.system("./gol_bits_seq 0 {} {} >> gol_bits_seq_results.out".format(d, rounds))
  os.system("./gol_seq 0 {} {} >> gol_seq_results.out".format(d, rounds))
