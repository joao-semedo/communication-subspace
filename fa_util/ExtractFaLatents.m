function [Z, U, Q, qOpt] = ExtractFaLatents(X, q)
% 
% Z = ExtractFaLatents(X, q) fits a Factor analysis model with latent
% dimensionality q and returns the orthonormalized latents Z. If q is a
% vector of latent dimensionalities ExtractFaLatents will first
% cross-validate the Factor Analysis model for each latent dimensionality
% in q and determine the optimal latent dimensionality among these. If q
% is not given, ExtractFaLatents will also perform the cross-validation
% procedure, with q = 0:p-1 (which can be slow).
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
% Z    - latent variables (N x q)
% U    - factor analysis dominant dimensions (p x q). These are obtained
% though the orthogonalization of the factor analysis loading matrix L:
% L = U*S*V'; The reconstructed data, under the factor analysis model, is
% then given by Z*U' + M, where M is the sample mean.
% Q    - factor analysis "decoding" matrix, i.e., Z = (X - M)*Q, where M
% is the sample mean (p x q)
% qOpt - optimal dimensionality found via cross-validation (1 x 1)
% qOpt = q if a latent dimensionality is provided)
% 
% @ 2018 Joao Semedo -- joao.d.semedo@gmail.com

C_CV_NUM_FOLDS = 10;
C_CV_OPTIONS = statset('crossval');

[n, p] = size(X);

if nargin < 2
	q = 0:p-1;
end

if numel(q) > 1
	qOpt = FactorAnalysisModelSelect...
		( CrossValFa(X, q, C_CV_NUM_FOLDS, C_CV_OPTIONS), q );
else
	qOpt = q;
end

Sigma = cov(X);
[L, psi] = FactorAnalysis( Sigma, qOpt );

Psi = diag(psi);

C = L*L' + Psi;

[U, S, V] = svd(L, 0);

Q = C\L*V*S';

m = mean(X);
M = m( ones(n, 1), : );

Z = (X - M)*Q;

end
