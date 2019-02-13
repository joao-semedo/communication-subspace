function [lambda, dof] = GetRidgeLambda(dMaxShrink, X, varargin)
% 
% lambda = GetRidgeLambda(dMaxShrink, X) computes an appropriate
% range for the Ridge regression regularization parameter lambda based on
% the source data X. This output can then be used to cross-validate Ridge
% regression. (See Elements of Statistical Learning, by Hastie, Tibshirani
% and Friedman for more information.)
% 
%   p: source data dimensionality
%   N: number of data points
% 
% INPUTS:
% 
% X - data matrix (N x p)
% 
% OUTPUTS:
% 
% lambda - Set of regularization paraments for Ridge regression. (1 x 51)
% 
% dof    - Effective degrees of freedom of a Ridge regression model
% applied to source data X with the regularization parameters given by
% lambda. (1 x 51)
% 
% OPTIONAL ARGUMENTS (NAME-VALUE PAIRS):
% 
% 'Scale' - 'True' (default) or "False'. Use variance scaling (z-scoring).
% 
% @ 2018 Joao Semedo -- joao.d.semedo@gmail.com

scale = true;
for i = 1:2:numel(varargin)
    switch upper(varargin{i})
        
        case 'SCALE'
            scale = varargin{i+1};
            
    end
end

m = mean(X,1);
s = std(X,0,1);
idxs = find( abs(s) < sqrt( eps(class(s)) ) );
if any(idxs)
    X(:,idxs) = [];
    m(idxs) = [];
	s(idxs) = [];
end

[n, K] = size(X);

M = m( ones(n, 1), : );
S = s( ones(n, 1), : );

if scale
    Z = (X - M) ./ S;
else
    Z = (X - M);
end

d = eig(Z'*Z);
dMax = max(d);
lambda = dMax*(1 - dMaxShrink) ./ dMaxShrink;

numLambdas = numel(lambda);
D = repmat(d, [1 numLambdas]);
Lambda = repmat(lambda, [K 1]);
dof = sum( D ./ (D + Lambda), 1 );

end
