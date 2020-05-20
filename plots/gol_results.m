close all

size = 2.^(3:16);

seq                 = [1.947E-06 2.104E-05 9.166E-05 1.594E-04 6.286E-04 2.850E-03 1.066E-02 3.643E-02 1.105E-01 4.500E-01 1.768E+00 8.185E+00 3.199E+01 1.187E+02];
seq_bit             = [1.852E-06 8.068E-06 3.783E-05 1.393E-04 2.577E-04 1.192E-03 5.280E-03 1.669E-02 7.764E-02 2.846E-01 9.084E-01 3.663E+00 1.459E+01 5.851E+01];
parallel_bpc_1t     = [1.206E-04 5.457E-04 1.361E-03 3.775E-03 1.418E-02 5.295E-02 2.204E-01 8.490E-01 3.394E+00 1.358E+01 5.433E+01 2.173E+02];
parallel_bpc_2t     = [9.564E-05 2.029E-04 5.395E-04 1.799E-03 6.936E-03 2.760E-02 1.104E-01 4.412E-01 1.766E+00 7.067E+00 2.827E+01 1.131E+02];
parallel_bpc_8t     = [2.631E-05 4.898E-05 1.495E-04 5.106E-04 1.959E-03 7.778E-03 2.810E-02 1.124E-01 8.908E-02 3.538E-01 1.414E+00 5.660E+00 22.63920229 90.57214796];
parallel_bpc_32t    = [3.701E-05 6.135E-05 1.571E-04 5.375E-04 1.842E-03 7.345E-03 2.927E-02 2.367E-02 9.186E-02 3.654E-01 2.820E+00 5.852E+00 2.346E+01];
parallel_bpc_128t   = [2.717E-05 6.425E-05 1.598E-04 5.133E-04 2.070E-03 7.714E-03 6.835E-03 2.457E-02 9.575E-02 3.831E-01 1.534E+00 6.186E+00];
parallel_bpc_512t   = [2.975E-05 6.966E-05 1.741E-04 1.003E-03 2.267E-03 2.546E-03 7.595E-03 2.840E-02 1.126E-01 4.811E-01 2.070E+00];
parallel_bpc_2048t  = [6.630E-05 1.568E-04 5.447E-04 2.042E-03 1.503E-03 2.794E-03 8.074E-03 2.914E-02 1.184E-01 8.376E-01];
parallel_bpc_500htt = [1.056E-03 1.076E-03 1.110E-03 1.087E-03 1.083E-03 8.142E-04];
parallel_bpc_2mt    = [5.178E-04 9.730E-04 3.210E-03 1.092E-02 6.745E-02];

parallel_lut_8t    = [4.920E-05 1.400E-04 5.110E-04 1.974E-03 7.813E-03 2.949E-02 1.135E-01 4.550E-01 1.818E+00 7.256E+00 2.904E+01 1.162E+02];
parallel_lut_64t   = [1.990E-05 3.620E-05 8.600E-05 2.670E-04 1.007E-03 3.984E-03 1.540E-02 5.726E-02 2.299E-01 9.181E-01 3.664E+00 1.477E+01];
parallel_lut_128t  = [2.080E-05 2.750E-05 5.590E-05 1.400E-04 5.190E-04 1.847E-03 7.308E-03 2.899E-02 1.158E-01 4.639E-01 1.874E+00 7.556E+00];
parallel_lut_512t  = [2.510E-05 3.360E-05 6.290E-05 1.810E-04 5.590E-04 2.113E-03 8.352E-03 3.315E-02 1.364E-01 5.996E-01 3.434E+00];
parallel_lut_2048t = [3.700E-05 6.580E-05 1.820E-04 5.850E-04 2.199E-03 9.930E-04 1.759E-03 5.082E-03 3.177E-02 2.087E-01];
parallel_lut_2mt   = [4.550E-04 8.620E-04 3.091E-03 1.114E-02 7.958E-02];


seq_bit_spdup             = seq        ./seq_bit;
parallel_bpc_1t_spdup     = seq(1:12)  ./parallel_bpc_1t;
parallel_bpc_2t_spdup     = seq(1:12)  ./parallel_bpc_2t;
parallel_bpc_8t_spdup     = seq        ./parallel_bpc_8t;
parallel_bpc_32t_spdup    = seq(2:end) ./parallel_bpc_32t;
parallel_bpc_128t_spdup   = seq(3:end) ./parallel_bpc_128t;
parallel_bpc_512t_spdup   = seq(4:end) ./parallel_bpc_512t;
parallel_bpc_2048t_spdup  = seq(5:end) ./parallel_bpc_2048t;
parallel_bpc_500htt_spdup = seq(9:end) ./parallel_bpc_500htt;
parallel_bpc_2mt_spdup    = seq(10:end)./parallel_bpc_2mt;

parallel_lut_8t_spdup    = seq( 3:end) ./ parallel_lut_8t;
parallel_lut_64t_spdup   = seq( 3:end) ./ parallel_lut_64t;
parallel_lut_128t_spdup  = seq( 3:end) ./ parallel_lut_128t;
parallel_lut_512t_spdup  = seq( 4:end) ./ parallel_lut_512t;
parallel_lut_2048t_spdup = seq( 5:end) ./ parallel_lut_2048t;
parallel_lut_2mt_spdup   = seq(10:end) ./ parallel_lut_2mt;

                 % seq time
bpc_perf_8192  = [1.768E+00	5.433E+01	2.827E+01	1.414E+00	7.047E-01	3.654E-01	1.828E-01	9.575E-02	5.125E-02	2.840E-02	2.848E-02	8.074E-03	1.084E-03	1.088E-03	1.087E-03	1.073E-03	1.085E-03	1.084E-03	1.085E-03	1.110E-03	1.110E-03	9.730E-04	1.421E-03];
bpc_perf_16384 = [8.185E+00	2.173E+02	1.131E+02	5.660E+00	2.820E+00	2.820E+00	7.345E-01	3.831E-01	2.039E-01	1.126E-01	1.120E-01	2.914E-02	1.075E-03	1.044E-03	1.070E-03	1.076E-03	1.103E-03	1.112E-03	9.188E-04	1.087E-03	1.089E-03	3.210E-03	2.833E-03];
bpc_perf_32768 = [3.199E+01				            2.264E+01	1.129E+01	5.852E+00	2.931E+00	1.534E+00	8.367E-01	4.811E-01	4.561E-01	1.184E-01	1.087E-03	9.361E-04	1.162E-03	1.162E-03	1.069E-03	1.099E-03	2.237E-04	1.083E-03	1.077E-03	1.092E-02	1.093E-02];
bpc_perf_65536 = [1.187E+02				            9.057E+01	4.517E+01	2.346E+01	1.177E+01	6.186E+00	3.395E+00	2.070E+00	1.932E+00	8.376E-01	8.696E-04	2.122E-04	1.055E-03	1.083E-03	9.584E-04	6.667E-04	2.155E-04	1.103E-03	1.091E-03	6.745E-02	1.453E-01];

lut_perf_8192  = [1.818E+00	                        2.299E-01	1.158E-01	6.021E-02	3.315E-02	3.390E-02	1.759E-03	6.890E-04	7.690E-04	6.950E-04	4.700E-04	7.210E-04	8.040E-04	7.510E-04	7.180E-04	7.210E-04	8.620E-04	1.707E-03];			
lut_perf_16384 = [7.256E+00	                        9.181E-01	4.639E-01	2.438E-01	1.364E-01	1.359E-01	5.082E-03	6.830E-04	7.400E-04	7.330E-04	7.320E-04	7.280E-04	7.170E-04	7.150E-04	4.930E-04	7.260E-04	3.091E-03	5.445E-03	3.068E-03	1.760E-05];	
lut_perf_32768 = [2.904E+01	                        3.664E+00	1.874E+00	1.076E+00	5.996E-01	5.405E-01	3.177E-02	7.270E-04	7.280E-04	6.140E-04	7.310E-04	7.320E-04	7.280E-04	7.420E-04	7.390E-04	7.270E-04	1.114E-02	1.090E-02	1.987E-02	1.000E-04	5.200E-05];
lut_perf_65536 = [1.162E+02	                        1.477E+01	7.556E+00	4.370E+00	3.434E+00	2.848E+00	2.087E-01	7.270E-04	7.280E-04	7.840E-04	8.250E-04	7.390E-04	7.450E-04	7.360E-04	6.810E-04	7.810E-04	7.958E-02	0.000594	0.00073  	0.045502	0.079578];

num_threads = 2.^[0:1 3:22];
bpc_spdup_8192  = bpc_perf_8192(1)  ./ bpc_perf_8192;
bpc_spdup_16384 = bpc_perf_16384(1) ./ bpc_perf_16384;
bpc_spdup_32768 = bpc_perf_32768(1) ./ bpc_perf_32768;
bpc_spdup_65536 = bpc_perf_65536(1) ./ bpc_perf_65536;

num_threads_2 = 2.^[3 6:25];
lut_spdup_8192  = bpc_perf_8192(1)  ./ lut_perf_8192;
lut_spdup_16384 = bpc_perf_16384(1) ./ lut_perf_16384;
lut_spdup_32768 = bpc_perf_32768(1) ./ lut_perf_32768;
lut_spdup_65536 = bpc_perf_65536(1) ./ lut_perf_65536;

%% Plot 
% close all
figure
set(gcf, 'Position',  [100, 100, 450, 400])
loglog(size        , seq                ,'linewidth', 2)
hold on
% grid on; 
grid minor
loglog(size        , seq_bit            ,'linewidth', 2)
loglog(size        , parallel_bpc_8t    ,'linewidth', 2)
loglog(size(3:end) , parallel_bpc_128t  ,'linewidth', 2)
loglog(size(4:end) , parallel_bpc_512t  ,'linewidth', 2)
loglog(size(5:end) , parallel_bpc_2048t ,'linewidth', 2)
loglog(size(10:end), parallel_bpc_2mt   ,'linewidth', 2)
% title("Runtime vs. World Size",'FontSize',15)
legend("Sequential", "Sequential Bit/Cell", ...
    '8 threads', '128 threads', '512 threads',...
    '2048 threads', ...
    '2 million threads', 'location',"Northwest")
xlabel("Size of World (NxN)")
ylabel("Average Time per Round")
saveas(gcf,'figures/bp_runtime_world.png')

%% Plot
% close all
figure
set(gcf, 'Position',  [100, 100, 450, 400])
loglog(size, ones(length(size),1), 'linewidth',2)
hold on, grid on
loglog(size        , seq_bit_spdup            ,'linewidth', 2)
loglog(size        , parallel_bpc_8t_spdup    ,'linewidth', 2)
loglog(size(3:end) , parallel_bpc_128t_spdup  ,'linewidth', 2)
loglog(size(4:end) , parallel_bpc_512t_spdup  ,'linewidth', 2)
loglog(size(5:end) , parallel_bpc_2048t_spdup ,'linewidth', 2)
loglog(size(10:end), parallel_bpc_2mt_spdup   ,'linewidth', 2)
% title("Speedup vs. World Size",'FontSize',15)
legend("Sequential", "Sequential Bit/Cell", '8 threads', ...
    '128 threads', '512 threads', '2048 threads', ...
    '2 million threads', 'location',"Northwest")
xlabel("Size of World (NxN)")
ylabel("Speedup (vs. Sequential)")
saveas(gcf,'figures/bp_speedup_world.png')

%% Plot
% close all
figure
set(gcf, 'Position',  [100, 100, 450, 400])
loglog(num_threads, bpc_spdup_8192(2:end), 'linewidth', 2)
hold on, grid on
loglog(num_threads, bpc_spdup_16384(2:end), 'linewidth', 2)
loglog(num_threads(3:end), bpc_spdup_32768(2:end),'linewidth', 2)
loglog(num_threads(3:end), bpc_spdup_65536(2:end),'linewidth', 2)
% title("Speedup vs. Number of Threads",'FontSize',15)
legend("N = 8192", 'N = 16384', 'N = 32768', 'N = 65536', 'location',"Southeast")
xlabel("Number of Threads")
ylabel("Speedup (vs. Sequential)")
saveas(gcf,'figures/bp_speedup_threads.png')

%% Plot runtime of 3x6 LUT w/ various numbers of threads
% close all
figure
set(gcf, 'Position',  [100, 100, 450, 400])
loglog(size        , seq                ,'linewidth', 2)
hold on, grid on
loglog(size( 3:end), parallel_lut_8t    ,'linewidth', 2)
loglog(size( 3:end), parallel_lut_128t  ,'linewidth', 2)
loglog(size( 4:end), parallel_lut_512t  ,'linewidth', 2)
loglog(size( 5:end), parallel_lut_2048t ,'linewidth', 2)
loglog(size(10:end), parallel_lut_2mt   ,'linewidth', 2)
% title("Runtime vs. World Size (3x6 LUT)",'FontSize',15)
legend("Sequential", "8 threads", '128 threads', '512 threads',...
    '2048 threads', '2 million threads', 'location',"Northwest")
xlabel("Size of Grid (NxN)")
ylabel("Average Time per Round")
saveas(gcf,'figures/lut3x6_runtime_world.png')

%% Plot speedup of 3x6 LUT w/ various numbers of threads
%close all
figure
set(gcf, 'Position',  [100, 100, 450, 400])
loglog(size, ones(length(size),1), 'linewidth',2)
hold on, grid on
loglog(size(3:end) , parallel_lut_8t_spdup  ,'linewidth', 2)
loglog(size(3:end) , parallel_lut_128t_spdup  ,'linewidth', 2)
loglog(size(4:end) , parallel_lut_512t_spdup  ,'linewidth', 2)
loglog(size(5:end) , parallel_lut_2048t_spdup ,'linewidth', 2)
loglog(size(10:end), parallel_lut_2mt_spdup   ,'linewidth', 2)
% title("Speedup vs. World Size (3x6 LUT)",'FontSize',15)
legend("Sequential", '8 threads', '128 threads', '512 threads', '2048 threads', ...
    '2 million threads', 'location',"Northwest")
xlabel("Size of Grid (NxN)")
ylabel("Speedup (vs. Sequential)")
saveas(gcf,'figures/lut3x6_speedup_world.png')

%% Plot
% close all
figure
set(gcf, 'Position',  [100, 100, 450, 400])
loglog(num_threads_2(1:18), lut_spdup_8192, 'linewidth', 2)
hold on, grid on
loglog(num_threads_2(1:20), lut_spdup_16384, 'linewidth', 2)
loglog(num_threads_2(1:21), lut_spdup_32768, 'linewidth', 2)
loglog(num_threads_2(1:21), lut_spdup_65536, 'linewidth', 2)
% title("Game of Life Speedup")
legend("N = 8192", 'N = 16384', 'N = 32768', 'N = 65536', 'location',"Southeast")
xlabel("Number of Threads")
ylabel("Speedup (vs. Sequential)")
saveas(gcf,'figures/lut3x6_speedup_threads.png')


%% Compare all 4 algos
% close all
figure
set(gcf, 'Position',  [100, 100, 450, 400])
loglog(size        , ones(length(size),1)     , 'linewidth',2)
hold on, grid on
loglog(size        , seq_bit_spdup            ,'linewidth', 2)
loglog(size(5:end) , parallel_bpc_2048t_spdup ,'linewidth', 2)
loglog(size(10:end), parallel_bpc_2mt_spdup   ,'linewidth', 2)
loglog(size(5:end) , parallel_lut_2048t_spdup ,'linewidth', 2)
loglog(size(10:end) , parallel_lut_2mt_spdup   ,'linewidth', 2)
% title("Game of Life Speedup Comparison")
legend("Sequential", 'Sequential Bit/Cell', 'Parallel Bit/Cell - 2048 threads', ...
    'Parallel Bit/Cell - 2 mil threads', '3x6 LUT - 2048 threads', ...
    '3x6 LUT - 2 mil threads', 'location',"Northwest")
xlabel("Size of Grid (NxN)")
ylabel("Speedup (vs. Sequential)")
saveas(gcf,'figures/speedup_comparison.png')
