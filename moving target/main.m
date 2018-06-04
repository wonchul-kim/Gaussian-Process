% clear 
% close all 

% observation
X_o = [ 1 1; 1 5; 1 9; 
        5 1; 5 5; 5 9;
        9 1; 9 5; 9 9; ]';

% Get distacne for P
[t, target, prediction_x] = target_position();
P0 = 150;
np = 4;
d0 = 0.5;
P = @(d) P0 - 10 * np * log(d / d0) + 5 * randn(1,1);
distance = @(x1, x2) sqrt((x1(1,1) - x2(1,1))^2 + (x1(2,1) - x2(2,1))^2);

% variables
sigma_n = 0.2;
sigma_f = 1.1251;
l = 5;

% kernel function 
error_function = @(x1, x2) sigma_n^2 * (sum(x1 == x2) == length(x1));
kernel_function = @(x1, x2) sigma_f^2 * exp((x1-x2)' * (x1-x2) / (-2 * l^2));
kernel = @(x1, x2) kernel_function(x1, x2) + error_function(x1, x2);

for n = 1:length(t)
    for i = 1:length(X_o)
        Y_o(i, n) = P(distance(X_o(:, i), target(:, n)));
    end
end

% % prediction_x
grid_size = size(prediction_x, 2);
% grid_size = 30;
[x1 x2] = meshgrid(1:grid_size, 1:grid_size);
prediction_x = [x1(:)'; x2(:)']./2;
% 

% 


% Kd
Kd = zeros(size(X_o, 2), size(X_o, 2)); 
for i = 1:size(X_o,2)
    for j = 1:size(X_o,2)
        Kd(i, j) = kernel(X_o(:, i), X_o(:, j));
    end
end
% Kd = Kd + triu(Kd, 1)';

% Kp
Kp = zeros(size(prediction_x, 2), size(prediction_x, 2));
for i = 1:size(prediction_x, 2)
    for j = i:size(prediction_x, 2)
        Kp(i, j) = kernel_function(prediction_x(:, i), prediction_x(:, j));
    end
end
% Kp = Kp + triu(Kp, 1)';

% Kpd
Kpd = zeros(size(prediction_x, 2), size(X_o, 2));
for i = 1:size(prediction_x, 2)
    for j = 1:size(X_o, 2)
        Kpd(i, j) = kernel_function(prediction_x(:, i), X_o(:, j));
    end
end
     
% mean & covariance
for n = 1:length(t)
    mu(:, n) = Kpd / Kd * Y_o(:, n); % mean ~ 0 
    % u = (Kpd / Kd * Y_o);   
    cov(:, n) = 1.96 * sqrt(diag(Kp - Kpd/Kd * Kpd')); 
    mu_grid(:, :, n) = reshape(mu(:, n), grid_size, grid_size); 
end     

figure();
prediction_x_grid = reshape(prediction_x(1,:),grid_size,grid_size);
prediction_y_grid = reshape(prediction_x(2,:),grid_size,grid_size);
xlabel('x'); ylabel('y'); zlabel('RSSI');
hold on;

% storage_mu_grid = mu_grid;
% storage_coord = zeros(length(t), 2);
coord_ = zeros(length(t), 2);
for n = 1:length(t)
    surf(prediction_x_grid, prediction_y_grid, mu_grid(:, :, n));
    surf(prediction_x_grid, prediction_y_grid, storage_mu_grid(:, :, n));

    disp(n);
    hold on;

    coord = zeros(2,1);
    [temp1,temp2] = max(mu_grid(:, :, n));
    [amplitude,coord(2)] = max(temp1);
    coord(1) = temp2(coord(2));
    coord = [prediction_x_grid(coord(2),coord(1)); prediction_y_grid(coord(2),coord(1))];
    % coord = coord/grid_size;
    disp('maximum value'); disp(amplitude);
    % disp('coordinate'); disp(coord.*grid_size);
    disp('coordinate'); disp(coord);
    % set(stem3(coord(2)*grid_size, coord(1)*grid_size, amplitude, 'r.'), 'MarkerSize', 30);
    set(stem3(coord(2), coord(1), 200, 'r.'), 'MarkerSize', 20);
    set(stem3(coord(2), coord(1), 200, 'ro'), 'MarkerSize', 15);
    coord_(n, :) = [ coord(2), coord(1) ];
    set(stem3(storage_coord(n, 2), storage_coord(n, 1), 200, 'b.'), 'MarkerSize', 20);
    set(stem3(storage_coord(n, 2), storage_coord(n, 1), 200, 'bo'), 'MarkerSize', 15);
%     storage_coord(n,:) = [coord(2), coord(1)];
%     set(plot(coord(2), coord(1), 'r.'), 'MarkerSize', 20);
    pause(1);
    hold on;
end
%%
pause(2);
figure(2);
for n = 1:length(t)
    set(plot(coord_(n, 1), coord_(n, 2), 'r.'), 'MarkerSize', 20);
    set(plot(coord_(n, 1), coord_(n, 2), 'ro'), 'MarkerSize', 20);
    hold on;
    set(plot(storage_coord(n, 1), storage_coord(n, 2), 'b.'), 'MarkerSize', 20);
    set(plot(storage_coord(n, 1), storage_coord(n, 2), 'bo'), 'MarkerSize', 20);
    hold on;

    xlim([0 20]);
    ylim([0 20]);
    pause(1);
    hold on;
end


% this gives a poor model, because we aren't using good parameters to model 
% the function. In order to get better parameters, we can maximize evidence 
% evidence = exp((Y_o'/Kd*Y_o+log(det(Kd))+length(Y_o)*log(2*pi))/-2);                          
% title (['this plot has evidence ' num2str(evidence)]) 
% legend('confidence bounds','mean','data points','location','SouthEast') 