function [B, B_, V] = ReducedRankRegress(Y, X, dim, varargin)
% 
% B = ReducedRankRegress(Y, X, dim) fits a Reduced Rank Regression model,
% with number of predictive dimensions given by dim, to target variables Y
% and source variables X, returning the mapping matrix B (which includes
% an intercept). If dim is a vector containing multiple numbers of
% predictive dimensions to be tested, ReducedRankRegress(Y, X, dim)
% returns an extended mapping matrix, containing the mapping matrices
% corresponding to each number of predictive dimensions tested.
% 
%   K:       target data dimensionality
%   p:       source data dimensionality
%   numDims: numbers of predictive dimensions to be tested
%   N:       number of data points
% 
% INPUTS:
% 
% Y   - target data matrix (N x K)
% X   - data matrix (N x p)
% dim - vector containing the numbers of predictive dimensions to be
% tested (1 x numDims)
% 
% OUTPUTS:
% 
% B  - Extended mapping matrix. B consists of the horizontal concatenation
% of each fitted mapping matrix (one for each number of predictive
% dimensions tested). As a result, and due to the included intercept, B
% has dimensions (p+1 x K*numDims). Predictions are obtained using: 
% Yhat = [ones(size(X,1),1) X]*B, where Yhat contains the models'
% predictions. Alternatively, [~, Yhat] = RegressPredict(Y, X, B).
% 
% B_ - Predictive dimensions: B_ = Bols*V, where the columns of V contain
% the eigenvectors of the optimal linear predictor Yhat = X*Bols (Bols is
% the ordinary least squared solution). The columns of B_, the
% predictive dimensions, are ordered according to target variance
% explained. For example, the top two predictive dimensions are B_(:,1:2).
% B_ is always of size p x K. The correct number of predictive dimensions
% should be found via cross-validation.
% 
% V  - The columns of V contain the eigenvectors of the optimal linear
% predictor Yhat = X*Bols (Bols is the ordinary least squared solution).
% 
% OPTIONAL ARGUMENTS (NAME-VALUE PAIRS):
% 
% 'RidgeInit' - 'True' (default) or "False'. Use Lambda-Reduced Rank
%               Regression.
% 'Scale'     - 'True' (default) or "False'. Use variance scaling
%               (z-scoring).
% 
% @ 2018 Joao Semedo -- joao.d.semedo@gmail.com

useRidgeInit = false;
scale = false;
for i = 1:2:numel(varargin)
    switch upper(varargin{i})
        
        case 'RIDGEINIT'
            C_RIDGE_D_MAX_SHRINKAGE_FACTOR = .5:.01:1;
            useRidgeInit = varargin{i+1};
            
        case 'SCALE'
            scale = varargin{i+1};
            
    end
end

% Exclude neurons with 0 variance.
m = mean(X,1);
s = std(X,0,1);
idxs = find( abs(s) < sqrt( eps(class(s)) ) );
if any(idxs)
    
    auxP = size(X, 2);
    auxIdxs = (1:auxP)';
    auxIdxs(idxs) = [];
    
    X(:,idxs) = [];
    m(idxs) = [];
    
end

[n, K] = size(Y);
p = size(X, 2);

M = m( ones(n, 1), : );
Z = (X - M);

if useRidgeInit
    lambda = GetRidgeLambda(C_RIDGE_D_MAX_SHRINKAGE_FACTOR, X, ...
        'Scale', scale);
    lambdaOpt = RegressModelSelect(@RidgeRegress, Y, X, lambda, ...
        'Scale', scale);
    Bfull = RidgeRegress(Y, X, lambdaOpt, 'Scale', scale);
    Bfull = Bfull(2:end,:);
else
    Bfull = Z\Y;
end

Yhat = Z*Bfull;
V = pca(Yhat);
B_ = Bfull*V;

% If any neurons were excluded, adjust B_ to the correct size.
if any(idxs)
	
	auxB_ = B_;
	
	B_ = zeros(auxP, K);
	B_(auxIdxs,:) = auxB_;
	
end

if nargin < 3
	B = [];
	return
end

if dim(1) == 0
    B = zeros(p, K);
else
    B = Bfull*V( :, 1:dim(1) )*V( :, 1:dim(1) )';
end

numDims = numel(dim);
if numDims > 1
    B(end,K*numDims) = 0;
    for i = 2:numDims
        B(:,K*(i-1)+1:K*i) = Bfull*V( :, 1:dim(i) )*V( :, 1:dim(i) )';
    end
end

B = [ repmat( mean(Y,1), [1 numDims] )-m*B; B ];

% If any neurons were excluded, adjust B to the correct size.
if any(idxs)
    
    auxIdxs = [1; auxIdxs+1];
    auxB = B;
    
    B = zeros(auxP+1, K*numDims);
    B(auxIdxs,:) = auxB;
    
end

end
