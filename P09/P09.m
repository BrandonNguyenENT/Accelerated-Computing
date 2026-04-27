data = readmatrix("p09_results.txt");

% Columns from CSV
density_percent = data(:,1);
density_fraction = data(:,2);

nnz_a = data(:,3);
nnz_b = data(:,4);

rocsp_time = data(:,5);
rocblas_time = data(:,6);

% Use density_fraction for log scale plots (safer than percent)
x = density_fraction;

figure;

loglog(x, rocsp_time, '-o');
hold on;
loglog(x, rocblas_time, '-o');

xlabel("Density (fraction)");
ylabel("Time (ms)");
legend("rocSPARSE", "rocBLAS");
grid on;
title("SpGEMM vs Dense GEMM Performance");