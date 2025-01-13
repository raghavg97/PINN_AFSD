% k\frac{d2T}{dx^2} - Q = 0 ; T(0) = T(1) = 300; Q = 1; k=0.01

%This code works
%Volumetric Integrations gave the appropriate equations. 
T_left = 300;
T_right = 400;

n = 100; %Number of cells

T = 200*ones(n,1); % Five "cells" and the temperature values at the cell centers.


del_x = 1/n;
Q = 5;
k = 0.01;
%BOundaries need to be different. 
tic
T_old = T;
T(1) = (T(2) + 2*T_left- Q*(del_x^2)/k)/3;
T(n) = (2*T_right + T(n-1) - Q*(del_x^2)/k)/3;
T(2:n-1) = (T(3:n) + T(1:n-2))/2 - Q*(del_x)^2/(2*k);
T_new = T;
iters = 0;
% for i=1:5000
while(sum(abs(T_old - T_new))>1e-5)
T_old = T;
T(1) = (T(2) + 2*T_left- Q*(del_x^2)/k)/3;
T(n) = (2*T_right + T(n-1) - Q*(del_x^2)/k)/3;
T(2:n-1) = (T(3:n) + T(1:n-2))/2 - Q*(del_x)^2/(2*k);
T_new = T;
iters=iters+1;
end
toc



%True Solution = 50 x^2 -50x + 300

x_fvm = linspace(del_x/2,1-del_x/2,n);

x_full = linspace(0,1,n);

T_true = @(x) (Q/(2*k))*x.^2 - (Q/(2*k) + (T_left - T_right))*x + T_left;


plot(x_fvm,T,'b'); hold on;
plot(x_full,T_true(x_full),'r--')