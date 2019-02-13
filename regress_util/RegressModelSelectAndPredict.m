function [loss, B] = RegressModelSelectAndPredict...
	(regressFun, Ytrain, Xtrain, Ytest, Xtest, alpha, varargin)
% 
% [loss, B] = RegressModelSelectAndPredict...
% 	(regressFun, Ytrain, Xtrain, Ytest, Xtest, alpha, varargin) fits
% regression model regressFun to training target and source data (Ytrain
% and Xtrain) and then predicts test target data Ytest using test source
% data Xtest and the model fit to the training data. Model selection is
% performed using 10-fold cross-validation on the training data, i.e.,
% the model used for test data prediction is fit to the training data
% using optimal regression parameter alphaOpt.
% (RegressModelSelectAndPredict is designed for use in nested
% cross-validation.)
% 
%   K:      target data dimensionality
%   p:      source data dimensionality
%   numPar: numbers of regularization parameters to be tested
%   Ntrain: number of training data points
%   Ntest:  number of testing data points
% 
% INPUTS:
% 
% regressFun - regression function handle. For example, regressFun =
% @RidgeRegress
% 
% Ytrain     - training target data matrix (Ntrain x K)
% Xtrain     - training source data matrix (Ntrain x p)
% Ytest      - testing target data matrix (Ntest x K)
% Xtest      - testing source data matrix (Ntest x p)
% alpha      - vector containing the regression parameters to be tested
% (1 x numPar)
% varargin   - additional parameters to be passed to either regressFun or
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
% @ 2018 Joao Semedo -- joao.d.semedo@gmail.com

lossMeasure = 'NSE';
i = 1;
while i < numel(varargin)
	switch upper(varargin{i})
		
	case 'LOSSMEASURE'
		lossMeasure = varargin{i+1};
		varargin(i:i+1) = [];
	
	otherwise
		i = i + 2;
	
	end
end

alphaOpt = RegressModelSelect...
	(regressFun, Ytrain, Xtrain, alpha, varargin{:});

B = regressFun(Ytrain, Xtrain, alphaOpt, varargin{:});

loss = RegressPredict(Ytest, Xtest, B, 'LossMeasure', lossMeasure);

end
