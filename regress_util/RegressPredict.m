function [loss, Yhat] = RegressPredict(Y, X, B, varargin)
% 
% loss = RegressPredict(Y, X, B) predicts target data Y using source data
% X and the mapping matrix B, and computes the error incurred.
% 
%   K: target data dimensionality
%   p: source data dimensionality
%   N: number of data points
% 
% INPUTS:
% 
% Y        - target data matrix (N x K)
% X        - source data matrix (N x p)
% B        - mapping matrix (p x K)
% varargin - additional parameters to be passed to either regressFun or
% RegressPredict (Name-Value pairs)
% 
% OUTPUTS:
% 
% loss - Loss incurred when predicting the test target data Ytest using
% the test source data Xtest and the model fit to the training data Ytrain
% and Xtrain, with optimal regression parameter alphaOpt. (1 x 1)
% 
% B    - Mapping matrix obtained from fitting regressFun, with optimal
% regression parameter alphaOpt, to the training data.
% 
% OPTIONAL ARGUMENTS (NAME-VALUE PAIRS):
%
% 'LossMeasure' - 'NSE' (Normalized Squared Error; default) or 'MSE' (Mean
% Squared Error)
% 
% @ 2018 Joao Semedo -- joao.d.semedo@gmail.com

lossMeasure = 'NSE';
for i = 1:2:numel(varargin)
	switch upper(varargin{i})
	
	case 'LOSSMEASURE'
		lossMeasure = varargin{i+1};
	
	end
end

Yhat = [ones(size(X,1),1) X]*B;

switch upper(lossMeasure)
case 'MSE'
	loss = MeanSquaredError(Y, Yhat);
case 'NSE'
	loss = NormalizedSquaredError(Y, Yhat);
case 'MVNSE'
	loss = MvNormalizedSquaredError(Y, Yhat);
end

end
