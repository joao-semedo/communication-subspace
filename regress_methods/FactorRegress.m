function B = FactorRegress(Y, X, q, varargin)
% 
% B = FactorRegress(Y, X, q) fit a Factor Regression model, with
% dimensionality given by q, to target variables Y and source variables
% X, returning the mapping matrix B (which includes an intercept). If q is
% a vector containing multiple latent dimensionalities to be tested, 
% FactorRegress(Y, X, q) returns an extended mapping matrix, containing
% the mapping matrices corresponding to each latent dimensionality tested.
% 
%   K:       target data dimensionality
%   p:       source data dimensionality
%   numDims: latent dimensionalities to be tested
%   N:       number of data points
% 
% INPUTS:
% 
% Y - target data matrix (N x K)
% X - data matrix (N x p)
% q - vector containing the latent dimensionalities to be tested (1 x
% numDims)
% 
% OUTPUTS:
% 
% B - Extended mapping matrix. B consists of the horizontal concatenation
% of each fitted mapping matrix (one for each latent dimensionality
% tested). As a result, and due to the included intercept, B has
% dimensions (p+1 x K*numDims). Predictions are obtained using: 
% Yhat = [ones(size(X,1),1) X]*B, where Yhat contains the models'
% predictions. Alternatively, [~, Yhat] = RegressPredict(Y, X, B).
% 
% OPTIONAL ARGUMENTS (NAME-VALUE PAIRS):
% 
% 'qOpt' - Optimal factor analysis dimensionality for the source activity
% X. When absent, Factor Regression will automatically determine qOpt via 
% cross-validation (which can be slow).
% 
% @ 2018 Joao Semedo -- joao.d.semedo@gmail.com

[n, K] = size(Y);
p = size(X, 2);

qOpt = [];
for i = 1:2:numel(varargin)
	switch upper(varargin{i})
	
	case 'QOPT'
		qOpt = varargin{i+1};
	
	end
end
if isempty(qOpt)
	qFactorAnalysis = 0:p-1;
	qOpt = FactorAnalysisModelSelect( ...
		CrossValFa(X, qFactorAnalysis), qFactorAnalysis);
end

m = mean(X);
M = m( ones(n, 1), : );

if qOpt == 0
    B = zeros(p, K);
    B = [ mean(Y,1); B ];
    return
end

q(q > qOpt) = [];
if isempty(q)
    q = qOpt;
end

Sigma = cov(X, 1);

s = diag(Sigma);
idxs = find( abs(s) < sqrt( eps(class(s)) ) );
if any(idxs)
% 	warning('%i silent units ignored', numel(idxs))
	auxP = size(X, 2);
	X(:,idxs) = [];
	Sigma(idxs,:) = [];
	Sigma(:,idxs) = [];
	auxIdxs = (1:auxP)';
	auxIdxs(idxs) = [];
end

[L, psi] = FactorAnalysis( Sigma, qOpt, varargin{:} );

Psi = diag(psi);

C = L*L' + Psi;

[~, S, V] = svd(L, 0);

Q = C\L*V*S';

if q(1) == 0
    B = zeros(p, K);
else
    EZ = (X - M)*Q(:,1:q(1));
    
    B = Q(:,1:q(1)) * (EZ\Y);
end

numDims = numel(q);
if numDims > 1
	B(end,K*numDims) = 0;
	for i = 2:numDims
		
		EZ = (X - M)*Q(:,1:q(i));
		
		B( :, K*(i-1)+1:K*i ) = Q(:,1:q(i)) * (EZ\Y);
		
	end
end

if any(idxs)
	auxB = B;
	B = zeros(auxP, K*numDims);
	B(auxIdxs,:) = auxB;
	
	auxM = m;
	m = zeros(1,auxP);
	m(auxIdxs) = auxM;
end

B = [ repmat( mean(Y,1), [1 numDims] ) - m*B; B ];

end
