function [cvLoss, cvLogLike] = CrossValFa(X, q, cvNumFolds, cvOptions)
% 
% cvLoss = CrossValFa(X, q) performs cross-validation for the Factor
% Analysis model. It computes the cross-validated log-likelihood for the
% data X and latent state dimensionalities q, selects the latent
% dimensionality qMax for which the cross-validated log-likelihood is
% highest and returns the cumulative shared variance explained by the
% latent dimensions under this model (FA with qMax dimensions).
% 
%   p: data dimensionality
%   q: latent dimensionality
%   N: number of data points
% 
% INPUTS:
% 
% X - data matrix (N x p)
% q - vector containing the latent dimensionalities to be tested (1 x
% numDims)
% 
% OUTPUTS:
% 
% cvLoss - cumulative shared variance explained by the latent dimensions
% for the FA model for which the cross-validated log-likehood is highest
% (1 x numDims)
% cvLogLike - cross-validated log-likelihood (numFolds x numDims)
% 
% OPTIONAL ARGUMENTS:
% 
% cvNumFolds - number of folds to be used when cross-validating
% cvOptions  - cross-validation options (type 'help statset' for more
%	info)
%
% @ 2018 Joao Semedo -- joao.d.semedo@gmail.com

if nargin < 3
    cvNumFolds = 10;
    cvOptions = statset;
end

q = sort(q);

cvFun = @(Xtrain, Xtest) FactorAnalysisTestLogLike(Xtrain, Xtest, q);

cvLogLike = crossval(cvFun, X, ...
    'KFold', cvNumFolds, ...
    'Options', cvOptions);

S = cov(X, 1);

[~, qMaxIdx] = max( nanmean(cvLogLike) );
qMax = q(qMaxIdx);
if qMax == 0
    cvLoss = NaN;
else
    L = FactorAnalysis(S, qMax);
    d = sort( eig( L*L' ), 'descend' );
    
    cvLoss = ( 1 - cumsum(d)/sum(d) )';
    
    if q(1) == 0
        cvLoss = [1 cvLoss];
        cvLoss = cvLoss(q + 1);
    else
        cvLoss = cvLoss(q);
    end
end

end

