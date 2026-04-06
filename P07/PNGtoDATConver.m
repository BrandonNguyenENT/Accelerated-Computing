clc; clear;

I = imread('peppers.png');
[m,n,c] = size(I);

fileID = fopen('peppers.dat','w');

for i = 1:m
    for j = 1:n
        for k = 1:c
            fprintf(fileID,'%d\n', I(i,j,k));
        end
    end
end

fclose(fileID);