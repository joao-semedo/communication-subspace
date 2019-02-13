function mse = MeanSquaredError(Ytest, Yhat)
% 
% mse = MeanSquaredError(Ytest, Yhat) computes the mean squared error
% between test data Ytest and predictions Yhat. Yhat may contain the
% predictions obtained from multiple models.
% 
%   K:         data dimensionality
%   numModels: number of prediction models used
%   N:         number of data points
% 
% INPUTS:
% 
% Ytest - test data matrix (N x K)
% Yhat  - predictions data matrix (N x K*numModels)
% 
% OUTPUTS:
% 
% mse - Mean squared error between the test data Ytest and the predictions
% from each model (1 x numModels).
% 
% @ 2018 Joao Semedo -- joao.d.semedo@gmail.com

[n, K] = size(Ytest);

numModels = size(Yhat, 2)/K;

squaredDif = bsxfun(@minus, repmat(Ytest, [1 numModels]), Yhat).^2;

mse = sum( reshape( sum( squaredDif, 1 )', [K numModels] ), 1)/(n*K);

end
