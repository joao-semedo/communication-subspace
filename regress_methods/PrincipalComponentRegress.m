function B = PrincipalComponentRegress(Y, X, dim, varargin)
% 
% B = PrincipalComponentRegress(Y, X, q) fits a Principal Component
% Regression model, with dimensionality given by q, to target variables
% Y and source variables X, returning the mapping matrix B (which includes
% an intercept). If q is a vector containing multiple dimensionalities to
% be tested, PrincipalComponentRegress(Y, X, dim) returns an extended
% mapping matrix, containing the mapping matrices corresponding to each
% dimensionality tested.
% 
%   K:       target data dimensionality
%   p:       source data dimensionality
%   numDims: numbers of principal components to be tested
%   N:       number of data points
% 
% INPUTS:
% 
% Y - target data matrix (N x K)
% X - data matrix (N x p)
% q - vector containing the numbers of principal components to be tested
% (1 x numDims)
% 
% OUTPUTS:
% 
% B - Extended mapping matrix. B consists of the horizontal concatenation
% of each fitted mapping matrix (one for each number of principal
% components tested). As a result, and due to the included intercept, B
% has dimensions (p+1 x K*numDims). Predictions are obtained using:
% Yhat = [ones(size(X,1),1) X]*B, where Yhat contains the models'
% predictions. Alternatively, [~, Yhat] = RegressPredict(Y, X, B).
% 
% @ 2018 Joao Semedo -- joao.d.semedo@gmail.com

[n, K] = size(Y);

m = mean(X,1);
s = std(X,0,1);
idxs = find( abs(s) < sqrt( eps(class(s)) ) );
if any(idxs)
	s(idxs) = 1;
end

M = m( ones(n, 1), : );
S = s( ones(n, 1), : );

Z = (X - M) ./ S;
if any(idxs)
	Z(:,idxs) = 1;
end

V = pca(Z);

B = V( :,1:dim(1) )*( ( Z * V( :,1:dim(1) ) ) \ Y );

numDims = numel(dim);
if numDims > 1
	B(end,K*numDims) = 0;
	for i = 2:numDims
		B( :, K*(i-1)+1:K*i ) ...
			= V( :,1:dim(i) )*( ( Z * V( :,1:dim(i) ) ) \ Y );
	end
end

B = B ./ repmat(s', [1 K*numDims]);
B = [ repmat( mean(Y,1), [1 numDims] )-m*B; B ];

end
