s = 4096;
%Pretend this is on the interval [0,32pi)
x = (1:N)'/N;
u = GRF1(N/2, 0, 2, 7, 7^2, "periodic");
u = u(x);

[uu, tt] = KS(u, 32*pi, 100, 1000, 0.01);

surf(tt,x,uu');
view([90 -90]); 
shading interp; 
colormap jet;
axis tight;
colorbar;

