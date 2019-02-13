function B = Regress(Y, X, varargin)
% 
% B = Regress(Y, X) fits a standard Linear Regression model to target
% variables Y and source variables X, returning the mapping matrix B
% (which includes an intercept).
% 
%   K: target data dimensionality
%   p: source data dimensionality
%   N: number of data points
% 
% INPUTS:
% 
% Y - target data matrix (N x K)
% X - data matrix (N x p)
% 
% OUTPUTS:
% 
% B  - Mapping matrix. Due to the included intercept, B has dimensions
% (p+1 x K). Predictions are obtained using: 
% Yhat = [ones(size(X,1),1) X]*B, where Yhat contains the model
% predictions. Alternatively, [~, Yhat] = RegressPredict(Y, X, B).
% 
% @ 2018 Joao Semedo -- joao.d.semedo@gmail.com

B = RidgeRegress(Y, X, 0);

end
