clear all; clc; close all;

%% gaussian process regression
%% initialize observations 
% X_o = [-1.5 -1 -0.75 -0.4 -0.3 0]'; 
% Y_o = [-1.6 -1.3 -0.5 0 0.3 0.6]'; 
[X_o, Y_o, prediction_x] = generateData_cos();

%% variables for kernel function
sigma_f_range = 0.01:0.04:4; 
l_range = 0.01:0.04:2;
sigma_n = 0.2;
sigma_f = 1.1251;
l = 0.90441; 

% miximize hyperparameter 
[sigma_f, l] = getHyperParameter(sigma_f_range, l_range, sigma_n, X_o, Y_o);

% kernel function 
error_function = @(x1, x2) sigma_n^2 * (x1 == x2);
kernel_function = @(x1, x2) sigma_f^2 * exp((x1-x2)^2 / (-2 * l^2));
kernel = @(x1, x2) kernel_function(x1, x2) + error_function(x1, x2);

% Kd
Kd = zeros(length(X_o),length(X_o)); 
for i = 1:length(X_o)
    for j = i:length(X_o)
        Kd(i, j) = kernel(X_o(i), X_o(j));
    end
end
Kd = Kd + triu(Kd, 1)';

% Kp
Kp = zeros(length(prediction_x), length(prediction_x));
for i = 1:length(prediction_x)
    for j = i:length(prediction_x)
        Kp(i, j) = kernel_function(prediction_x(i), prediction_x(j));
    end
end
Kp = Kp + triu(Kp, 1)';

% Kpd
Kpd = zeros(length(prediction_x), length(X_o));
for i = 1:length(prediction_x)
    for j = 1:length(X_o)
        Kpd(i, j) = kernel_function(prediction_x(i), X_o(j));
    end
end
    
% mean & covariance
u = (Kpd * inv(Kd) * Y_o); % mean ~ 0 
% u = (Kpd / Kd * Y_o);
cov = 1.96 * sqrt(diag(Kp - Kpd/Kd * Kpd'));    
    
figure
plot_variance = @(x,lower,upper,color) set(fill([x,x(end:-1:1)],[upper,fliplr(lower)],color),'EdgeColor',color); 
plot_variance(prediction_x, (u - cov)', (u + cov)', [0.8 0.8 0.8]) 
hold on
set(plot(prediction_x, u, 'k-'), 'Linewidth', 3);
set(plot(X_o, Y_o, 'r.'), 'MarkerSize', 15);

% this gives a poor model, because we aren't using good parameters to model 
% the function. In order to get better parameters, we can maximize evidence 
evidence = exp((Y_o'/Kd*Y_o+log(det(Kd))+length(Y_o)*log(2*pi))/-2);                          
title (['this plot has evidence ' num2str(evidence)]) 
legend('confidence bounds','mean','data points','location','SouthEast') 


    
    