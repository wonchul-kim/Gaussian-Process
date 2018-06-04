function [ t, target, prediction_x ] = target_position()


global dt T pred_dt
dt = 1;
pred_dt = 0.5;
T = 20;

t = 0.1:dt:T;
pred_t = 0.1:pred_dt:T;
% x(1, 1:10) = 25 - t(1, 1:10);
% y(1, 1:10) = 25 - t(1, 1:10) +randn();
% x(1, 11:20) = 25 - (2 .* t(1, 11:20));
% y(1, 11:20) = 25 - (2 .* t(1, 11:20));
x(1, 1:10) = t(1, 1:10);
y(1, 1:10) = t(1, 1:10) + abs(randn());
x(1, 11:20) = 2 .* t(1, 11:20) + 1;
y(1, 11:20) = 2 .* t(1, 11:20) - 1;


target = [x; y];
prediction_x = zeros(2, length(pred_t));

end

