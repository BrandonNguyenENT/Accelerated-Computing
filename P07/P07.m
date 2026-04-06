clc
close all
m = 384; % height
n = 512; % width
c = 3;   % channels
P = zeros(m,n,c);
fid = fopen('peppers.out','r');
i = 1; j = 1; k = 1;
for l = 1:m*n*c,
    p = fscanf(fid,'%d', 1);
    P(i,j,k) = p;    
    if mod(k,c) == 0
        k = 1;
        j = j + 1;
        if j == n + 1
            j = 1;
            i = i + 1;
            if i == m + 1
                break;
            end
        end
    else
        k = k + 1;
    end
end
fclose(fid);
[M,N] = meshgrid(1:1:n,1:1:m);
c = 1;
surf(M,N,P(:,:,c));
xlabel('Image Width')
ylabel('Image Height')
title(['Sobel 5x5 Convolution Output, channel ',num2str(c)])
grid on
grid minor
set(gcf,'color','w')
axis equal
colorbar
shading interp
rotate3d
view(0,90);
set(gcf,'position',[10,10,1280,768])
figure(1)