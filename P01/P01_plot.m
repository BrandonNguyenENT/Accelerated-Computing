close all
clc
data = load('data.txt');
threads = data(:,1);
times = data(:,2);
figure;
plot(threads, times, 'o-', 'LineWidth', 2);
set(gca, 'XTick', threads);
set(gca, 'XTickLabel', ...
    arrayfun(@(x) sprintf('2^{%d}', log2(x)), threads, 'UniformOutput', false));
xlabel('Hardware Threads', 'FontSize', 12);
ylabel('Elapsed (wall clock) time, s', 'FontSize', 12);
title('Programming Assignment 01 (OpenMP)', 'FontSize', 14);
set(gcf,'Color','white');
grid on;
ax = gca;
ax.YMinorGrid = 'on';
ax.XMinorGrid = 'on';

% ---- Find optimal (minimum) time ----
[min_time, idx] = min(times);       % find the minimum time
opt_threads = threads(idx);         % corresponding thread count

% ---- Mark the optimal point on the plot ----
hold on
plot(opt_threads, min_time, 'rp', 'MarkerSize', 12, 'MarkerFaceColor', 'r'); % red point
text(opt_threads, min_time, ...
    sprintf('Optimal: %d threads\n%.4f s', opt_threads, min_time), ...
    'FontSize', 11, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
hold off

figure(1);





