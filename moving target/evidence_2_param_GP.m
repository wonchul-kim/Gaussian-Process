function [ log_evidence ] = evidence_2_param_GP(x,p, X_o, Y_o) 
  sigma_n = p; 
  error_function = @(x,x2) sigma_n^2*(sum(x==x2) == length(x)) ;  
  parametrized_kernel_function = @(x,x2,p_sigma_f,p_l) p_sigma_f^2*exp((x-x2)' * (x - x2) / (-2*p_l^2)); 
  X_o = X_o'; 
  Y_o = Y_o; 
  K = zeros(size(X_o, 1), size(X_o, 1)); 
  for i=1:size(X_o, 1) 
      for j=1:size(X_o, 1) 
           K(i,j)=parametrized_kernel_function(X_o(i, :),X_o(j, :),x(1),x(2))+error_function(X_o(i, :),X_o(j, :)); 
       end 
   end 
   log_evidence=0.5*Y_o'/K*Y_o+log(det(K))+size(X_o, 1)/2*log(2*pi);%this is the negative of the log evidence, 
                                       %so this is to be minimized 
end 


% sigma_n = 0.2;% we know the amount of noise from the data 
% sigma_range = 0.01:0.04:4; 
% l_range = 0.01:0.04:2; 
% evidence=zeros(length(sigma_range),length(l_range)); 
% for i=1:length(sigma_range) 
%     for j=1:length(l_range) 
%        evidence(i,j)=evidence_2_param_GP([sigma_range(i) l_range(j)],sigma_n); 
%     end 
% end 