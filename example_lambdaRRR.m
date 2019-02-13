%%

SET_CONSTS

load('mat_sample/sample_data.mat')

%% lambda-Reduced Rank Regression cross-validation examples ++++++++++++++
% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

%% Cross-validate Reduced Rank Regression

% Vector containing the interaction dimensionalities to use when fitting
% RRR. 0 predictive dimensions results in using the mean for prediction.
numDimsUsedForPrediction = 0:25;

% Number of cross validation folds.
cvNumFolds = 10;

% Initialize default options for cross-validation.
cvOptions = statset('crossval');

% If the MATLAB parallel toolbox is available, uncomment this line to
% enable parallel cross-validation.
% cvOptions.UseParallel = true;

% Regression method to be used.
regressMethod = @ReducedRankRegress;

% Auxiliary function to be used within the cross-validation routine (type
% 'help crossval' for more information). Briefly, it takes as input the
% the train and test sets, fits the model to the train set and uses it to
% predict the test set, reporting the model's test performance. Here we
% are used NSE (Normalized Squared Error) as the performance metric. MSE
% (Mean Squared Error) is also available.
cvFun = @(Ytrain, Xtrain, Ytest, Xtest) RegressFitAndPredict...
	(regressMethod, Ytrain, Xtrain, Ytest, Xtest, ...
	numDimsUsedForPrediction, 'LossMeasure', 'NSE', ...
    'RidgeInit', false, 'Scale', false);
% RidgeInit is an extra argument for ReducedRankRegress. Extra arguments
% for the regression function are passed as name/value pairs after the
% cross-validation parameter (in this case numDimsUsedForPrediction).

% Cross-validation routine.
cvl = crossval(cvFun, Y_V2, X, ...
	  'KFold', cvNumFolds, ...
	'Options', cvOptions);

% Stores cross-validation results: mean loss and standard error of the
% mean across folds.
cvLoss = [ mean(cvl); std(cvl)/sqrt(cvNumFolds) ];

% To compute the optimal dimensionality for the regression model, call
% ModelSelect:
optDimReducedRankRegress = ModelSelect...
	(cvLoss, numDimsUsedForPrediction);

% Plot Reduced Rank Regression cross-validation results
x = numDimsUsedForPrediction;
y = 1-cvLoss(1,:);
e = cvLoss(2,:);

errorbar(x, y, e, 'o-', 'MarkerFaceColor', 'w', 'MarkerSize', 7)

xlabel('Number of predictive dimensions')
ylabel('Predictive performance')



%% Cross-validate Lambda-Reduced Rank Regression (no scaling)

numDimsUsedForPrediction = 0:25;

cvNumFolds = 10;

cvOptions = statset('crossval');
% cvOptions.UseParallel = true;

regressMethod = @ReducedRankRegress;

% To use Lambda-Reduced Rank Regression, set 'RidgeInit' to 1.
cvFun = @(Ytrain, Xtrain, Ytest, Xtest) RegressFitAndPredict...
	(regressMethod, Ytrain, Xtrain, Ytest, Xtest, ...
	numDimsUsedForPrediction, 'LossMeasure', 'NSE', ...
    'RidgeInit', true, 'Scale', false);

% Cross-validation routine.
cvl = crossval(cvFun, Y_V2, X, ...
	  'KFold', cvNumFolds, ...
	'Options', cvOptions);

% Stores cross-validation results: mean loss and standard error of the
% mean across folds.
cvLoss = [ mean(cvl); std(cvl)/sqrt(cvNumFolds) ];

% To compute the optimal dimensionality for the regression model, call
% ModelSelect:
optDimLambdaReducedRankRegress = ModelSelect...
	(cvLoss, numDimsUsedForPrediction);

% Plot Lambda-Reduced Rank Regression cross-validation results
x = numDimsUsedForPrediction;
y = 1-cvLoss(1,:);
e = cvLoss(2,:);

hold on
errorbar(x, y, e, 'o-', 'MarkerFaceColor', 'w', 'MarkerSize', 7)
hold off

xlabel('Number of predictive dimensions')
ylabel('Predictive performance')



%% Cross-validate Lambda-Reduced Rank Regression (scaled)

numDimsUsedForPrediction = 0:25;

cvNumFolds = 10;

cvOptions = statset('crossval');
% cvOptions.UseParallel = true;

regressMethod = @ReducedRankRegress;

% To use Lambda-Reduced Rank Regression, set 'RidgeInit' and 'Scale' to 1.
cvFun = @(Ytrain, Xtrain, Ytest, Xtest) RegressFitAndPredict...
	(regressMethod, Ytrain, Xtrain, Ytest, Xtest, ...
	numDimsUsedForPrediction, 'LossMeasure', 'NSE', ...
    'RidgeInit', true, 'Scale', true);

% Cross-validation routine.
cvl = crossval(cvFun, Y_V2, X, ...
	  'KFold', cvNumFolds, ...
	'Options', cvOptions);

% Stores cross-validation results: mean loss and standard error of the
% mean across folds.
cvLoss = [ mean(cvl); std(cvl)/sqrt(cvNumFolds) ];

% To compute the optimal dimensionality for the regression model, call
% ModelSelect:
optDimLambdaReducedRankRegressScaled = ModelSelect...
	(cvLoss, numDimsUsedForPrediction);

% Plot Lambda-Reduced Rank Regression cross-validation results
x = numDimsUsedForPrediction;
y = 1-cvLoss(1,:);
e = cvLoss(2,:);

hold on
errorbar(x, y, e, 'o-', 'MarkerFaceColor', 'w', 'MarkerSize', 7)
hold off

xlabel('Number of predictive dimensions')
ylabel('Predictive performance')



%%
legend('Reduced Rank Regression', ...
    '\lambda{}Reduced Rank Regression', ...
    '\lambda{}Reduced Rank Regression (Scaled)', ...
	'Location', 'SouthEast')


