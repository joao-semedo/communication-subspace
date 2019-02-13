function [alphaOpt, optLoss] = ModelSelect(cvLoss, alpha)
% 
% alphaOpt = ModelSelect(cvLoss, alpha) selects the optimal regression
% parameter alpha based on the cross-validated loss. The optimal
% parameter is chosen as the simplest model for which the performance is
% within 1 S.E.M. of the peak performance (across all models).
% 
%   numPar: number of regression parameters used
% 
% INPUTS:
% 
% cvLoss - Cross-validated loss. The cvLoss matrix contains two rows: the
% first row contains the mean (across folds) cross-validated loss for each
% regression parameter; the second row contains the standard error of
% the mean cross validated loss (across folds). (2 x numPar)
% 
% alpha  - Model parameters associated with each column in cvLoss. The
% model parameters in alpha must be ordered in terms of model complexity,
% i.e., from the simplest to the most complex model. For example, for
% Ridge regression, alpha is the regularization parameter lambda, which
% should be in decreasing order. For reduced rank regression, alpha is the
% number of predictive dimensions, which should be in increasing order.
% (1 x numPar)
% 
% OUTPUTS:
% 
% alphaOpt - Optimal regression parameter, chosen by taking the simplest
% model for which the test performance is within 1 S.E.M. of the peak test
% performance (across all models). (1 x 1)
% 
% optLoss  - Average test loss corresponding to the optimal parameter
% alphaOpt. (1 x 1)
% 
% @ 2018 Joao Semedo -- joao.d.semedo@gmail.com

loss = cvLoss(1,:);
[minLoss, minIdx] = min(loss);
stdErrorMinLoss = cvLoss(2,minIdx);

alphaOptIdx = find( loss <= minLoss + stdErrorMinLoss, 1 );
alphaOpt = alpha(alphaOptIdx);
optLoss = loss(alphaOptIdx);

end
