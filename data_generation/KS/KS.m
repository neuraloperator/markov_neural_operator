%u - initial condition
%l - length of interval [0,l) or [-l/2,l/2)
%T - final time 
%N - number of solutions to record
%h - internal time step
function [uu, tt] = KS(u, l, T, N, h)
s = length(u(:));

v = fft(u);

k = (2*pi/l)*[0:s/2-1 0 -s/2+1:-1]';
L = k.^2 - k.^4;
E = exp(h*L); E2 = exp(h*L/2);
M = 64;
r = exp(1i*pi*((1:M)-.5)/M);
LR = h*L(:,ones(M,1)) + r(ones(s,1),:);
Q = h*real(mean((exp(LR/2)-1)./LR,2));
f1 = h*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2));
f2 = h*real(mean((2+LR+exp(LR).*(-2+LR))./LR.^3,2));
f3 = h*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));

uu = zeros(N,s); 
tt = zeros(N,1);
nmax = round(T/h);
nrec = floor((T/N)/h);
g = -0.5i*k;
q = 1;
for n = 1:nmax
    t = n*h;
    Nv = g.*fft(real(ifft(v)).^2);
    a = E2.*v + Q.*Nv;
    Na = g.*fft(real(ifft(a)).^2);
    b = E2.*v + Q.*Na;
    Nb = g.*fft(real(ifft(b)).^2);
    c = E2.*a + Q.*(2*Nb-Nv);
    Nc = g.*fft(real(ifft(c)).^2);
    v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3;
    if mod(n,nrec)==0
        u = real(ifft(v));
        uu(q,:) = u; tt(q) = t;
        q = q + 1;
    end
    %disp(n);
end