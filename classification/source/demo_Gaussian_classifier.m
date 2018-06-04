function demo_classification  
% Generative classification demo: The Gaussian classifier with shared diagonal covariance.
% That is, a Gaussian Naive Bayes with shared covariance.
% Note, whenever the covariance is shared (i.e. the same for both classes) then the decision function is linear.

% This code is provided to complete Assignment 1 for the module Machine Learning (Extended)
% It (akong with Assignment 1) also serves as a starting point for you if you want to experiment with variations
% of the Gaussian Classifier such as non-diagonal class-covariance, non-shared class covariances etc.

% Generate some data in 2 classes
nInputs = 2;  % dimensionality of data (number of attributes)
nClasses = 2; % number of classes
nSamples(1) = 200; nSamples(2) = 200; % number of points in each class

mu1 = [1;-1]; sigma1 = 1.2;
mu2 = [-1;1]; sigma2 = 1.8;
X = zeros(nInputs,max(nSamples),nClasses);
X(:,1:nSamples(1),1) = sigma1*randn(nInputs,nSamples(1)) + ...
    repmat(mu1,1,nSamples(1)); % class 1 training data
X(:,1:nSamples(2),2) = sigma2*randn(nInputs,nSamples(2)) + ...
    repmat(mu2,1,nSamples(2)); % class 2 training data

% Alternatively, load data

% targets
Y = [1,0]; % targets for class 1,2.

% Plot the data
clf, figure(1)
plot(X(1,1:nSamples(1),1),X(2,1:nSamples(1),1),'b+'); hold on % class 1
plot(X(1,1:nSamples(1),2),X(2,1:nSamples(1),2),'ro'); % class 2
title('2-CLASS DATA')


disp('Press a key to continue training the GAUSSIAN CLASSIFIER model ...'); pause


N = sum(nSamples); % total number of points
Sigma = zeros(nInputs);
for c = 1:nClasses
    Nc = nSamples(c); Xc = X(:,1:Nc,c);
    alpha(c) = Nc/N; % class prior

    % estimate the mean of class c
    m(:,c) = sum(Xc,2)/Nc; % Equivalent to: mean(Xc')
    
   % estimate covariance of diagonal form for class c
   % Sigma(:,:,c) = diag(diag((Xc-repmat(m(:,c),1,Nc))*(Xc-repmat(m(:,c),1,Nc))'))/Nc; % each class with its own covariance
       % Equiv. to: cov(Xc')    
           
    Sigma = Sigma + (Xc-repmat(m(:,c),1,Nc))*(Xc-repmat(m(:,c),1,Nc))'; % instead, here we take a shared covariance
    % this is estimated from all the points, not just from class c.

end%for c

% ML if all classes have same variance
Sigma = diag(diag(Sigma))/N;  % diagonal approximation of the shared covariance estimate.
inv_Sigma = inv(Sigma);

figure(1)
plotgaus(m(:,1)',Sigma,'b'); 
plotgaus(m(:,2)',Sigma,'r');

% Computing & plotting the estimated decision boundary -- this is linear because the class covariances 
% were taken to be equal in the two classes. In case you decide to experiment with each class having its own covariance then 
% remove the following three lines.
delta_m = m(:,1)-m(:,2); mean_m = (m(:,1)+m(:,2))/2;
w = inv_Sigma*delta_m;
b = -delta_m'*inv_Sigma*mean_m + log(alpha(1)/(1-alpha(1)));

xmin = -3; xmax = 3; 
hl2 = line([xmin,xmax],[(-b-w(1)*xmin)/w(2),(-b-w(1)*xmax)/w(2)]);
set(hl2,'color','g','linewidth',2);
drawnow


function [h] = plotgaus( mu, sigma, colspec );

% PLOTGAUS Plotting of a Gaussian contour
%
%    PLOTGAUS(MU,SIGMA,COLSPEC) plots the mean and standard
%    deviation ellipsis of the Gaussian process that has mean MU
%    variance SIGMA, with color COLSPEC = [R,G,B].
%
%    If you use only PLOTGAUS(MU,SIGMA), the default color is
%    [0 1 1] (cyan).
%

if nargin < 3; colspec = [0 1 1]; end;
npts = 100;

stdev = sqrtm(sigma);

t = linspace(-pi, pi, npts);
t=t(:);

X = [cos(t) sin(t)] * stdev + repmat(mu,npts,1);

h(1) = line(X(:,1),X(:,2),'color',colspec,'linew',2);
h(2) = line(mu(1),mu(2),'marker','+','markersize',10,'color',colspec,'linew',2);


