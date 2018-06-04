function [t, x, y] = target( )
% moving target position

global dt T

dt = 0.05;
T = 10;

t = 0:dt:T;

x = 5 * cos(0.1 * t);
y = 5 * sin(0.1 * t);
end

