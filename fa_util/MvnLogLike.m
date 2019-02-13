function logLike = MvnLogLike(X, m, S)
% 
% logLike = MvnLogLike(X, m, S) Computes the log-likelihood of the data in
% X under a multivariate Gaussian distribution with mean m and covariance
% S
% 
%   p: data dimensionality
%   N: number of data points
% 
% INPUTS: 
%
% X - data matrix (N x p)
% m - mean (1 x p)
% S - covariance matrix (p x p)
%
% OUTPUTS:
%
% logLike - log likelihood
% 
% @ 2018 Joao Semedo -- joao.d.semedo@gmail.com

[N, p] = size(X);

M = m( ones(N, 1), : );
X = (X - M);

logLike = -(1/2)*( N*p*log(2*pi) + N*logdet(S) ...
	+ sum(sum( X .* ( S\X' )' )) );

end
