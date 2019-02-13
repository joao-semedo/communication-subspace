function B = RidgeRegress(Y, X, lambda, varargin)
% 
% B = RidgeRegress(Y, X, lambda) fits a Ridge Regression model, with
% regularization parameter lambda, to target variables Y and source
% variables X, returning the mapping matrix B (which includes an
% intercept). If lambda is a vector containing multiple regularization
% parameters to be tested, RidgeRegress(Y, X, lambda) returns an extended
% mapping matrix, containing the mapping matrices corresponding to each
% regularization parameter tested.
% 
%   K:      target data dimensionality
%   p:      source data dimensionality
%   numPar: numbers of regularization parameters to be tested
%   N:      number of data points
% 
% INPUTS:
% 
% Y      - target data matrix (N x K)
% X      - data matrix (N x p)
% lambda - vector containing the regularization parameters to be tested
% (1 x numPar)
% 
% OUTPUTS:
% 
% B  - Extended mapping matrix. B consists of the horizontal concatenation
% of each fitted mapping matrix (one for each regularization parameter
% tested). As a result, and due to the included intercept, B has
% dimensions (p+1 x K*numPar). Predictions are obtained using: 
% Yhat = [ones(size(X,1),1) X]*B, where Yhat contains the models'
% predictions. Alternatively, [~, Yhat] = RegressPredict(Y, X, B).
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

[n, K] = size(Y);
p = size(X, 2);

m = mean(X,1);
s = std(X,0,1);
idxs = find( abs(s) < sqrt( eps(class(s)) ) );
if any(idxs)
	s(idxs) = 1;
end

M = m( ones(n, 1), : );
S = s( ones(n, 1), : );

if scale
    Z = (X - M) ./ S;
else
    Z = X - M;
end

if any(idxs)
	Z(:,idxs) = 1;
end

Zplus = [Z; sqrt( lambda(1) )*eye(p)];
Yplus = [Y; zeros(p, K)];

B = Zplus\Yplus;

numLambdas = numel(lambda);
if numLambdas > 1
	B(end,K*numLambdas) = 0;
	for i = 2:numLambdas
		Zplus(end-p+1:end,:) = sqrt( lambda(i) )*eye(p);
		
		B( :, K*(i-1)+1:K*i ) = Zplus\Yplus;
	end
end

if scale
	B = B ./ repmat(s', [1 K*numLambdas]);
end
B = [ repmat( mean(Y,1), [1 numLambdas] )-m*B; B ];

end
