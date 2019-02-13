function [alphaOpt, cvLoss] ...
	= RegressModelSelect(regressFun, Y, X, alpha, varargin)
% 
% alphaOpt, cvLoss ...
% 	= RegressModelSelect(regressFun, Y, X, alpha, varargin) finds the
% 	optimal regression parameter alpha, for regression function
% 	regressFun, by performing 10-fold cross-validation on target data Y
% 	and source data X.
% 
%   K:      target data dimensionality
%   p:      source data dimensionality
%   numPar: numbers of regularization parameters to be tested
%   N:      number of data points
% 
% INPUTS:
% 
% regressFun - regression function handle. For example, regressFun =
% @RidgeRegress
% 
% Y          - target data matrix (N x K)
% X          - source data matrix (N x p)
% alpha      - vector containing the regression parameters to be tested
% (1 x numPar)
% varargin   - additional parameters to be passed to either regressFun or
% RegressPredict (Name-Value pairs)
% 
% OUTPUTS:
% 
% alphaOpt - Optimal regression parameter, chosen by taking the simplest
% model for which the test performance is within 1 S.E.M. of the peak test
% performance (across all models). (1 x 1)
% 
% optLoss  - Average test loss corresponding to the optimal parameter
% alphaOpt. (1 x 1)
% 
% @ 2018 Joao Semedo -- joao.d.semedo@gmail.com

C_CV_NUM_FOLDS = 10;
C_CV_OPTIONS = statset('crossval');

cvFun = @(Ytrain, Xtrain, Ytest, Xtest) RegressFitAndPredict...
	(regressFun, Ytrain, Xtrain, Ytest, Xtest, alpha, varargin{:});

cvLoss = crossval(cvFun, Y, X, ...
	  'KFold', C_CV_NUM_FOLDS, ...
	'Options', C_CV_OPTIONS);

cvLoss = [ mean(cvLoss); std(cvLoss)/sqrt(C_CV_NUM_FOLDS) ];

alphaOpt = ModelSelect(cvLoss, alpha);

end
