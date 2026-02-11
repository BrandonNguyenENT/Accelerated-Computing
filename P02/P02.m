close all
clc

data = load('data.txt');

% Extract columns
N      = data(:,1);
serial = data(:,2);
t2     = data(:,3);
t4     = data(:,4);
t8     = data(:,5);
t16    = data(:,6);
t32    = data(:,7);
t64    = data(:,8);
t128   = data(:,9);

figure;
hold on;

% Plot Serial (BLACK per requirement)
plot(N, serial, 'k-', 'LineWidth', 2);

% Plot Parallel Results (different colors automatically assigned)
plot(N, t2,   '-o', 'LineWidth', 2);
plot(N, t4,   '-o', 'LineWidth', 2);
plot(N, t8,   '-o', 'LineWidth', 2);
plot(N, t16,  '-o', 'LineWidth', 2);
plot(N, t32,  '-o', 'LineWidth', 2);
plot(N, t64,  '-o', 'LineWidth', 2);
plot(N, t128, '-o', 'LineWidth', 2);

% Log scale is appropriate for large N growth
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');

xlabel('Problem Size (N)', 'FontSize', 12);
ylabel('Runtime t (seconds)', 'FontSize', 12);
title('OpenMP Runtime vs Problem Size', 'FontSize', 14);

legend('Serial (n=1)', ...
       'n=2', 'n=4', 'n=8', 'n=16', 'n=32', 'n=64', 'n=128', ...
       'Location', 'northwest');

grid on;
set(gcf,'Color','white');

hold off;
