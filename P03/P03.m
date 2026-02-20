close all
clc

threads = [1, 2, 4, 8, 16, 32, 64, 128, 256];
times = [4.222591, 1.722158, 0.887656, 0.493156, 0.296353, 0.186523, 0.114281, 0.102343, 0.082594];

% Calculate Speedup for analysis
serial_time = times(1);
speedup = serial_time ./ times;

% Create Figure for Runtime
figure('Name', 'Runtime Analysis');
plot(threads, times, 'o-', 'LineWidth', 2, 'MarkerFaceColor', 'b', 'MarkerSize', 8);

% Set X-axis to display in 2^n format
set(gca, 'XTick', threads);
set(gca, 'XTickLabel', arrayfun(@(x) sprintf('2^{%d}', round(log2(x))), threads, 'UniformOutput', false));

% Labels and Title
xlabel('Hardware Threads', 'FontSize', 12);
ylabel('Elapsed (wall clock) time, s', 'FontSize', 12);
title('Programming Assignment 03 (OpenMP - Simpson''s 1/3 Rule)', 'FontSize', 14);

% Aesthetics
set(gcf, 'Color', 'white');
grid on;
ax = gca;
ax.YMinorGrid = 'on';
ax.XMinorGrid = 'on';

% Highlight the optimal point (16 threads)
[min_time, min_idx] = min(times);
hold on;
plot(threads(min_idx), min_time, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
text(threads(min_idx), min_time, sprintf(' Optimal: %.4fs (16 Threads)', min_time), ...
    'VerticalAlignment', 'bottom', 'FontWeight', 'bold');

% (Optional) Create Figure for Speedup
figure('Name', 'Speedup Analysis');
plot(threads, speedup, 's-', 'LineWidth', 2, 'Color', [0.8500 0.3250 0.0980]);
hold on;
plot(threads, threads, '--k'); % Ideal speedup line

set(gca, 'XTick', threads);
set(gca, 'XTickLabel', arrayfun(@(x) sprintf('2^{%d}', round(log2(x))), threads, 'UniformOutput', false));

xlabel('Hardware Threads');
ylabel('Speedup Factor');
title('Speedup Analysis (Relative to 1 Thread)');
legend('Measured Speedup', 'Ideal Speedup', 'Location', 'northwest');
grid on;
set(gcf, 'Color', 'white');

figure(1);
