
import os


dims      = [ 2**i for i in range(8, 17) ]
nthreads  = [ 2**i for i in range(8, 23) ]

rounds = 10

for d in dims:
  # os.system("echo running ./gol_seq 0 {} {} >> gol_seq_results.out".format(d))
  # os.system("./gol_bits_seq 0 {} {} >> gol_bits_seq_results.out".format(d, rounds))
  # os.system("./gol_seq 0 {} {} >> gol_seq_results.out".format(d, rounds))
  for p in nthreads:
    os.system("./gol_parallel 0 {} {} {} >> gol_parallel_results.out".format(d, rounds, p))
