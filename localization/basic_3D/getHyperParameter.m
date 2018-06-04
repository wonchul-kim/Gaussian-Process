function [ sigma_f, l ] = getHyperParameter( sigma_f_range, l_range, sigma_n, X_o, Y_o )

evidence = zeros(length(sigma_f_range), length(l_range));
for i = 1:length(sigma_f_range)
    for j = 1:length(l_range)
        evidence(i, j) = evidence_2_param_GP(sigma_f_range(i), l_range(j), sigma_n, X_o, Y_o);
    end
end

%get the max l and sigma 
[v location]=min(evidence(:)); 
l_location=floor(location/length(sigma_f_range))+1; 
sigma_location=mod(location,l_location*length(sigma_f_range)-length(sigma_f_range)); 
l = l_range(l_location); 
sigma_f = sigma_f_range(sigma_location); 

