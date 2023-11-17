s = 512;
%Pretend this is on the interval [0,2*pi*L)
x = (1:s)'/s;

N = 200; % number of case
T = 1000; % time
t = 10000; % time steps
u_out = zeros(N, t, s);

for i=1:N
    disp(i);
    u = GRF1(s/2, 0, 2, 5, 5^2, "periodic");
    u = u(x);

    [uu, tt] = KS(u, 2*pi*32, T, t, 0.1);
    u_out(i,:,:) = uu;

%     surf(tt,x,uu');
%     view([90 -90]); 
%     shading interp; 
%     colormap jet;
%     axis tight;
%     colorbar;
end



