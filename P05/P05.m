sizes = [64, 128, 256, 512, 1024, 2048, 4096];


basic_time = [0.011264, 0.013312, 0.026624, 0.133120, 0.888832, 6.980608, 66.746368];
tiled_time = [0.025600, 0.012288, 0.023552, 0.090112, 0.600064, 4.686848, 29.979649];


figure;
plot(sizes, basic_time, '-o', 'Color', 'b', 'LineWidth', 2, 'MarkerSize', 6); hold on;
plot(sizes, tiled_time, '-s', 'Color', 'r', 'LineWidth', 2, 'MarkerSize', 6);


set(gca, 'XScale', 'log', 'YScale', 'log');


xticks(sizes);
xticklabels(arrayfun(@num2str, sizes, 'UniformOutput', false));


xlabel('Matrix Size (Powers of 2)');
ylabel('Kernel Elapsed Time (ms)');
title('Kernel Elapsed Time Comparison: Basic vs Tiled');
legend('Basic Kernel', 'Tiled Kernel', 'Location', 'northwest');
grid on;


xlim([min(sizes) max(sizes)]);
ylim([min([basic_time, tiled_time])*0.8 max([basic_time, tiled_time])*1.2]);