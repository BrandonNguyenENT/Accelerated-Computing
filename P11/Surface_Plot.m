% =========================================================
% 3D Surface Plot Script
% =========================================================
% X-axis = Threads
% Y-axis = MPI Ranks (M)
% Z-axis = Wall Time (wall_s)
% =========================================================

clc;
clear;
close all;

% Load CSV file
T = readtable('p11_surface.csv');

% Unique values
threads_vals = unique(T.threads);
ranks_vals = unique(T.M);

% Create meshgrid
[X, Y] = meshgrid(threads_vals, ranks_vals);

% Initialize Z matrix
Z = zeros(length(ranks_vals), length(threads_vals));

% Fill Z matrix
for i = 1:length(ranks_vals)

    for j = 1:length(threads_vals)

        idx = (T.M == ranks_vals(i)) & ...
              (T.threads == threads_vals(j));

        Z(i,j) = T.wall_s(idx);
    end
end

% Create surface plot
figure;

surf(X, Y, Z);

% Improve appearance
shading interp;
colorbar;

xlabel('Threads');
ylabel('MPI Rank (M)');
zlabel('Wall Time (s)');

title('3D Surface Plot of Performance');

% Better viewing angle
view(135, 30);

% Logarithmic x-axis
set(gca, 'XScale', 'log');

xticks([32 64 128 256 512 1024]);
xticklabels({'32','64','128','256','512','1024'});