clc;
clear;
close all;

T = readtable('p11_surface.csv');

ranks = unique(T.M);

figure;
hold on;
grid on;

for i = 1:length(ranks)

    rank_val = ranks(i);

    idx = (T.M == rank_val);

    threads = T.threads(idx);
    walltime = T.wall_s(idx);

    [threads_sorted, order] = sort(threads);
    wall_sorted = walltime(order);

    plot(threads_sorted, wall_sorted, '-o', ...
        'LineWidth', 2, ...
        'DisplayName', sprintf('Rank %d', rank_val));
end

xlabel('Threads');
ylabel('Wall Time (s)');
title('Performance Comparison Across MPI Ranks');

legend('Location', 'best');

set(gca, 'XScale', 'log');

xticks([32 64 128 256 512 1024]);
xticklabels({'32','64','128','256','512','1024'});

hold off;