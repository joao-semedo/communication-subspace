function logLike = FactorAnalysisTestLogLike(Xtrain, Xtest, q, varargin)
% 
% logLike = FactorAnalysisTestLogLike(Xtrain, Xtest, q) applies factor
% analysis models with latent dimensionalities given by q to Xtrain, and
% computes the log-likelihood of the data in Xtest under these models.
% 
%   p:      data dimensionality
%   q:      latent dimensionality
%   Ntrain: size of training data set
%   Ntest:  size of testing data set
% 
% INPUTS: 
%
% Xtrain - train data matrix (Ntrain x p)
% Xtest  - test data matrix (Ntest x p)
% q - vector containing the latent dimensionalities to be tested (1 x
% numDims)
% 
% OUTPUTS:
%
% logLike - log likelihood of test data under the models fit to the
% training data (1 x numDims)
% 
% OPTIONAL ARGUMENTS (NAME-VALUE PAIRS):
%
% 'Method' - 'FA' (default) or 'PPCA'
%
% @ 2018 Joao Semedo -- joao.d.semedo@gmail.com

m = mean(Xtrain);
S = cov(Xtrain, 1);

numDims = numel(q);
logLike = zeros(1,numDims);
for i = 1:numDims
	
	if q(i) == 0
		Psi = diag( diag( S ) );
		
        [~,p] = chol(Psi);
        if p
            logLike(i) = NaN;
        else
            logLike(i) = MvnLogLike(Xtest, m, Psi);
        end
	else
		[L, psi] = FactorAnalysis( S, q(i), varargin{:} );
		
		idxs = find( abs(psi) < sqrt( eps(class(psi)) ) );
		if any(idxs)
			logLike(i) = NaN;
			continue
		end
		
		Psi = diag(psi);
		C = L*L' + Psi;
        
        [~,p] = chol(C);
        if p
            logLike(i) = NaN;
        else
            logLike(i) = MvnLogLike(Xtest, m, C);
        end
	end
	
end

end
