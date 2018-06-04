clear all;

train_data = [0 0;
    20 0;
    40 0;
    55 0;
    55 20;
    55 40;
    55 55;
    0 20;
    0 40;
    0 55;
    20 55;
    40 55];

% y_data = [0.2;
%     0.5;
%     0.7;
%     0.9;
%     1.3;
%     0.1;
%     0.2;
%     0.3;
%     0.8;
%     0.5;
%     1;
%     0.9];
target = [10 20]';
[; y_data] = virtual_node(target, train_data);
% train_data = node_coord;
% y_data = Rssi;

sigma_f = 10;
l =  10; 
kernel_function = @(x1,x2) sigma_f^2*exp(((x1-x2)'*(x1-x2))/(-2*l^2)); 

sigma_n = 0;
error_function = @(x1,x2) sigma_n^2*(x1==x2);

k = @(x1,x2) kernel_function(x1,x2) + error_function(x1,x2);

grid_size = 30;
[x,y] = meshgrid(0:grid_size,0:grid_size);
arbitrary_x = [x(:)';y(:)']*2;

% Kd
k_prior = zeros(length(train_data(:,1)),length(train_data(:,1)));
for i=1:length(train_data(:,1))
    for j=1:length(train_data(:,1))
        k_prior(i,j) = kernel_function(train_data(i,:)',train_data(j,:)');
        k_prior(i,j) = k_prior(i,j) + error_function(i,j);
    end
end

% Kpd
k_s = zeros(length(arbitrary_x(1,:)),length(train_data(:,1)));
for i=1:length(arbitrary_x(1,:))
    for j=1:length(train_data(:,1))
        k_s(i,j) = kernel_function(arbitrary_x(:,i),train_data(j,:)');
    end
end

% Kp
K = zeros((grid_size+1)^2,(grid_size+1)^2);
for i=1:(grid_size+1)^2
    for j=1:(grid_size+1)^2
        K(i,j) = kernel_function(arbitrary_x(:,i),arbitrary_x(:,j));
    end
end
nn = length(k_prior);
Mu = (k_s*inv(k_prior+0.01*eye(nn,nn)))*y_data;
Sigma = 1.96*sqrt(diag(K-k_s/k_prior*k_s'));
Mu_tran = reshape(Mu,1+grid_size,1+grid_size)';

% [V,D]=eig(K); 
% A=V*(D.^(1/2)); 

figure(); hold on;
plot3(train_data(:,1),train_data(:,2),y_data,'o');
x_coord = reshape(arbitrary_x(1,:),1+grid_size,1+grid_size);
y_coord = reshape(arbitrary_x(2,:),1+grid_size,1+grid_size);
surf(x_coord,y_coord,Mu_tran);
xlabel('x'); ylabel('y'); zlabel('z');

coord = zeros(2,1);
[temp1,temp2] = max(Mu_tran);
[amplitude,coord(2)] = max(temp1);
coord(1) = temp2(coord(2));
coord = [x_coord(coord(2),coord(1)); y_coord(coord(2),coord(1))];
coord = coord/grid_size;

disp('maximum value'); disp(amplitude);
disp('coordinate'); disp(coord);

