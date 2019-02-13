function qOpt = FactorAnalysisModelSelect(cvLoss, q)
% 
% qOpt = FactorAnalysisModelSelect(cvLoss, q) selects the optimal
% dimensionality for a factor analysis model based on the cumulative
% shared variance explained by the latent dimensions (cvLoss). q contains
% the number of latent dimensions corresponding to each entry in cvLoss.
% 
% The optimal dimensionality is selected as the minimum number of latent
% dimensions necessary to account for 95% of the shared variance, as
% defined by the factor analysis model for which the cross-validated
% log-likelihood was highest (See Williamson et al., PLOS Computational
% Biology, 2016)
% 
% INPUTS:
% 
% cvLoss - cumulative shared variance explained by the latent dimensions
% for the FA model for which the cross-validated log-likehood is highest
% (1 x numDims)
% q      - vector containing the latent dimensionalities corresponding to each
% entry in cvLoss (1 x numDims)
% 
% OUTPUTS:
% 
% qOpt - optimal dimensionality for the factor analysis model
% 
% @ 2018 Joao Semedo -- joao.d.semedo@gmail.com

VAR_TRESHOLD = .95;   % Use VAR_TRESHOLD = f to select the minimum number
%						of latent dimensions necessary to account for
%						f*100% of the shared variance, as defined by the
%						factor analysis model for which the
%						cross-validated log-likelihood was highest. Use
%						VAR_TRESHOLD = 1 - eps to select the latent
%						dimensionality for which the cross-validated
%						log-likelihood was highest.

if isnan(cvLoss)
    qOpt = 0;
else
    qOpt = q( find( 1-cvLoss > VAR_TRESHOLD, 1 ) );
end

end
