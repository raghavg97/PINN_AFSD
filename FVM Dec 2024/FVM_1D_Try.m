% k\frac{d2T}{dx^2} - Q = 0 ; T(0) = T(1) = 300; Q = 1; k=0.01
%Volumetric Integrations gave the appropriate equations. 
T_left = 300;
T_right = 500;

n = 50000; %Number of cells

T = 200*ones(n,1); % Five "cells" and the temperature values at the cell centers.


del_x = 1/n;
Q = 0;
k = 0.01;
%BOundaries need to be different. 
T_old = T;
T(1) = (T(2) + 4*T_left- 2*Q*(del_x^2)/k)/5;
T(n) = (4*T_right + T(n-1) - 2*Q*(del_x^2)/k)/5;
T(2:n-1) = (T(3:n) + T(1:n-2))/2 - Q*(del_x)^2/k;
T_new = T;
% for i=1:5000
while(sum(abs(T_old - T_new))>1e-1)
T_old = T;
T(1) = (T(2) + 4*T_left- 2*Q*(del_x^2)/k)/5;
T(n) = (4*T_right + T(n-1) - 2*Q*(del_x^2)/k)/5;
T(2:n-1) = (T(3:n) + T(1:n-2))/2 - Q*(del_x)^2/k;
T_new = T;
end



%True Solution = 50 x^2 -50x + 300

x = linspace(del_x/2,1-del_x/2,n);

T_true = @(x) (Q/(2*k))*x.^2 - (Q/(2*k) + (T_left - T_right))*x + T_left;


plot(T,'b'); hold on;
plot(T_true(x),'r')