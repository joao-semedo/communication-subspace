function nse = NormalizedSquaredError(Ytest, Yhat)
% 
% nse = NormalizedSquaredError(Ytest, Yhat) computes the normalized
% squared error between test data Ytest and predictions Yhat. Yhat may
% contain the predictions obtained from multiple models.
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
% nse - Normalized squared error between the test data Ytest and the
% predictions from each model (1 x numModels). Normalized squared error is
% given by the sum of squared errors divided by the total sum of squares
% of the target data.
% 
% @ 2018 Joao Semedo -- joao.d.semedo@gmail.com

% nse = 1 - r^2

K = size(Ytest, 2);

numModels = size(Yhat, 2)/K;

squaredDif = bsxfun(@minus, repmat(Ytest, [1 numModels]), Yhat).^2;

rss = sum( reshape( sum( squaredDif, 1 )', [K numModels] ), 1);

N = size(Ytest, 1);
tss = sum( sum( ( Ytest - repmat( mean(Ytest), [N 1] ) ).^2 ) );

nse = rss/tss;

end
