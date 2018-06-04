function [ X_o, Y_o, prediction_x ] = generateData_cos()

%%
global train_dt pred_dt T
train_dt = 0.5;
pred_dt = 0.01;
T = 4*pi;

% tranning data
X_o = 0:train_dt:T;
Y_o = zeros(1, size(X_o, 2));
Y_o = cos(X_o) + 0.2 * randn(1, size(X_o, 2));
% Y_o = cos(X_o);

% predcition data for x
prediction_x = 0:pred_dt:T;

plot(X_o(1,:), Y_o(1, :), 'linewidth', 0.1, 'color', [0 0.5 0])
axis equal
grid on
title (['this is original data']) 


X_o = X_o';
Y_o = Y_o';

end

