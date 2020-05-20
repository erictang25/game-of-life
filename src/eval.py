
import os
import csv

# dims      = [ 2**i for i in range(4, 17) ]
# nthreads  = [ 2**i for i in range(6, 22) ]
dims      = [ 2**i for i in range(4, 17) ]
nthreads  = [ 8 ]

rounds = 15

def run_evals(output_file):
  for d in dims:
    # os.system("echo running ./gol_seq 0 {} {} >> gol_seq_results.out".format(d))
    # os.system("./gol_bits_seq 0 {} {} >> gol_bits_seq_results.out".format(d, rounds))
    # os.system("./gol_seq 0 {} {} >> gol_seq_results.out".format(d, rounds))
    for p in nthreads:
      if d*d/8 >= p:
        os.system("./gol_lut3x6_parallel 0 {} {} {} >> {}".format(d, rounds, p, output_file))
        os.system("./gol_parallel 0 {} {} {} >> {}".format(d, rounds, p, output_file))


def write_to_csv(input_file, output_file):
  data_dict = {}
  with open(input_file, 'r') as fd:
    contents = fd.readlines()
    for j in contents:
      lst = j.split('|')
      type_  = int(lst[0])
      rounds = int(lst[1])
      N      = int(lst[2])
      P      = int(lst[3])
      time   = float(lst[4])
      if type_ not in data_dict:
        data_dict[type_] = {}
      if N not in data_dict[type_]:
        data_dict[type_][N] = {}
      data_dict[type_][N][P] = time
  # print (data_dict)
  with open(output_file, 'w' ) as csvfile:
    results = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for type_key, type_val in sorted(data_dict.items()):
      header = [type_key]
      header.extend([P_key for P_key in sorted(type_val[dims[-1]].keys())])
      
      results.writerow(header)
      for N_key, N_val in sorted(type_val.items()):
        t_threads = []
        for P_key, P_val in sorted(N_val.items()):
          t_threads.append( P_val )
        row = [N_key]
        row.extend(t_threads)
        results.writerow(row)


if __name__ == "__main__":
  sim_output = 'gol_parallel_results.out'
  csv_output = 'run_times.csv'
  run_evals(sim_output)
  write_to_csv(sim_output, csv_output)
