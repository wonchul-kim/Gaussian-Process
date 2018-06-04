clear 
close all 

% observation
X_o = [ 1 1; 1 5; 1 9; 
        5 1; 5 5; 5 9;
        9 1; 9 5; 9 9;  ];

% Get distacne for P
[t, x, y] = target();
target_position = zeros(length(t), 2);
for i = 1:length(t)
    target_position(i, :) = [x(i), y(i)];
end

P0 = 150;
np = 4;
d0 = 0.5;
P = @(d) P0 - 10 * np * log(d / d0) + 5 * randn(1,1);
distance = @(x1, x2) sqrt((x1(1,1) - x2(1,1))^2 + (x1(1,2) - x2(1,2))^2);

% variables
sigma_n = 0.2;
sigma_f = 1.1251;
l = 5; 

% kernel function 
error_function = @(x1, x2) sigma_n^2 * (sum(x1 == x2) == length(x1));
kernel_function = @(x1, x2) sigma_f^2 * exp((x1-x2)' * (x1-x2) / (-2 * l^2));
kernel = @(x1, x2) kernel_function(x1, x2) + error_function(x1, x2);

% prediction_x
grid_size = 30;
[x1 x2] = meshgrid(1:grid_size, 1:grid_size);
prediction_x = [x1(:)'; x2(:)']./3;

Kp = zeros(size(prediction_x, 2), size(prediction_x, 2), length(t));

figure(1);
hold on;
for n = 1:length(t)
    for i = 1:length(X_o)
        Y_o(i, n) = P(distance(X_o(i, :), target_position(n, :)));
    end
    
    X_o = X_o';

    % Kd
    Kd = zeros(size(X_o, 2), size(X_o, 2)); 
    for i = 1:size(X_o,2)
        for j = i:size(X_o,2)
            Kd(i, j) = kernel(X_o(:, i), X_o(:, j));
        end
    end
    Kd = Kd + triu(Kd, 1)';
 
   % Kp
    for i = 1:size(prediction_x, 2)
        for j = 1:size(prediction_x, 2)
            Kp(i, j, n) = kernel_function(prediction_x(:, i), prediction_x(:, j));
        end
    end
%     Kp = Kp + triu(Kp, 1)';
    
    % Kpd
    Kpd = zeros(size(prediction_x, 2), size(X_o, 2));
    for i = 1:size(prediction_x, 2)
        for j = 1:size(X_o, 2)
            Kpd(i, j) = kernel_function(prediction_x(:, i), X_o(:, j));
        end
    end
    
    % mean & covariance
    mu(:, n) = Kpd / Kd * Y_o(:, n); % mean ~ 0 
    % u = (Kpd / Kd * Y_o);
    cov(:, n) = 1.96 * sqrt(diag(Kp(:, :, n) - Kpd/Kd * Kpd')); 

    mu_grid = reshape(mu(:,n), grid_size, grid_size); 
   
    prediction_x_grid = reshape(prediction_x(1, :), grid_size, grid_size);
    prediction_y_grid = reshape(prediction_x(2, :), grid_size, grid_size);

    
    hold on
    surf(prediction_x_grid, prediction_y_grid, mu_grid);
    xlabel('x'); ylabel('y'); zlabel('RSSI'); hold on;
    coord = zeros(2,1);
    [temp1,temp2] = max(mu_grid);
    [amplitude,coord(2)] = max(temp1);
    coord(1) = temp2(coord(2));
    coord = [prediction_x_grid(coord(2),coord(1)); prediction_y_grid(coord(2),coord(1))];
    set(stem3(coord(2), coord(1), amplitude, 'r.'), 'MarkerSize', 10);
    hold on;
    disp('maximum value'); disp(amplitude);
    disp('coordinate'); disp(coord);
    disp(n);
    X_o = X_o';

end
    
% figure()
% hold on;
% if( n < 2)
%     prediction_x_grid = reshape(prediction_x(1,:),grid_size,grid_size);
%     prediction_y_grid = reshape(prediction_x(2,:),grid_size,grid_size);
%     surf(prediction_x_grid, prediction_y_grid, mu_grid);
%     xlabel('x'); ylabel('y'); zlabel('RSSI');
% 
%     coord = zeros(2,1);
%     [temp1,temp2] = max(mu_grid);
%     [amplitude,coord(2)] = max(temp1);
%     coord(1) = temp2(coord(2));
%     coord = [prediction_x_grid(coord(2),coord(1)); prediction_y_grid(coord(2),coord(1))];
%     set(stem3(coord(2), coord(1), amplitude, 'r.'), 'MarkerSize', 30);
%     hold on;
%     disp('maximum value'); disp(amplitude);
%     disp('coordinate'); disp(coord);
%     disp(n);

    % this gives a poor model, because we aren't using good parameters to model 
    % the function. In order to get better parameters, we can maximize evidence 
    % evidence = exp((Y_o'/Kd*Y_o+log(det(Kd))+length(Y_o)*log(2*pi))/-2);                          
    % title (['this plot has evidence ' num2str(evidence)]) 
    % legend('confidence bounds','mean','data points','location','SouthEast') 