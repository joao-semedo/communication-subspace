function loss = RegressFitAndPredict...
	(regressFun, Ytrain, Xtrain, Ytest, Xtest, alpha, varargin)
% 
% loss = RegressFitAndPredict...
% 	(regressFun, Ytrain, Xtrain, Ytest, Xtest, alpha, varargin) fits
% regression model regressFun (with regression parameters alpha) to
% training target and source data (Ytrain and Xtrain) and then predicts
% test target data Ytest using test source data Xtest and the model fit to
% the training data.
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
% and Xtrain, with the regression parameters in alpha. (1 x numPar)
% 
% @ 2018 Joao Semedo -- joao.d.semedo@gmail.com

B = regressFun(Ytrain, Xtrain, alpha, varargin{:});

loss = RegressPredict(Ytest, Xtest, B, varargin{:});

end
