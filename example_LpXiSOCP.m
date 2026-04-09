% =========================================================================
% Example: Basic Use of Lp_XiSOCP_L1cvx
% =========================================================================
%
% This script provides a minimal working example of how to train and
% evaluate the Lp-XiSOCP model using the IRL1 algorithm implemented in:
%
%   Lp_XiSOCP_L1cvx.m
%
% For simplicity, the same dataset is used here for both training and
% testing. Therefore, this script is intended only as a usage example and
% should not be interpreted as a proper final performance evaluation.
%
% For a rigorous model assessment, use an independent test set or perform
% cross-validation (see CV_Lp_XiSOCP.m).
%
% The user can choose how the matrices S1 and S2 are constructed:
%   'chol'  : covariance-based Cholesky factorization
%   'estim' : sample-based estimator
%
% See Lp_XiSOCP_L1cvx.m for further details on these two options.
% =========================================================================

clear; clc; close all;

addpath(genpath('./dataset_bin_FS'));
addpath(genpath('./Models'));

%% =========================
%  Load dataset
%  =========================
 load('colorectal.mat');   datasetName = 'Colorectal';
% load('lymphoma_XY.mat'); datasetName = 'Lymphoma';
% load('pomeroy.mat');     datasetName = 'Pomeroy';
% load('DermaMNIST_All_0vs2.mat'); datasetName='DermaMNIST0vs2';

[m,n] = size(X);
fprintf('Dataset: %s\n', datasetName);
fprintf('Samples: %d | Features: %d\n', m, n);

%% =========================
%  Set model parameters
%  =========================
FunPara.type = 'estim';    % 'estim' or 'chol'
FunPara.p    = 0.2;        % Lp quasi-norm exponent, 0 < p < 1
FunPara.c    = 2^6;        % regularization parameter
FunPara.epsi = 1e-5;       % smoothing parameter for IRL1

nu = [0.2; 0.8];           % confidence parameters
FunPara.kapa = sqrt(nu./ (1 - nu));

fprintf('\nModel parameters:\n');
fprintf('  type = %s\n', FunPara.type);
fprintf('  p    = %.2f\n', FunPara.p);
fprintf('  C    = %.4f\n', FunPara.c);
fprintf('  epsi = %.1e\n', FunPara.epsi);
fprintf('  Alpha = [%.4f, %.4f]\n', nu(1), nu(2));

%% =========================
%  Train and predict
%  =========================
% Here Xt = X is used only for demonstration purposes.
[Prediction, Sol] = Lp_XiSOCP_L1cvx(X, Y, X, FunPara);

%% =========================
%  Evaluate performance
%  =========================
[BACCU, ACCU, F1] = medi_auc_accu(Prediction, Y);

%% =========================
%  Display results
%  =========================
fprintf('\n=== Results on %s ===\n', datasetName);
fprintf('BACCU : %.4f\n', BACCU);
fprintf('ACCU  : %.4f\n', ACCU);
fprintf('F1    : %.4f\n', F1);
fprintf('NNZ   : %d\n', Sol.nnz);
fprintf('Iter  : %d\n', Sol.Iter);

if isfield(Sol,'TimeIter') && ~isempty(Sol.TimeIter)
    fprintf('CPU time (total): %.4f sec\n', sum(Sol.TimeIter));
end

if isfield(Sol,'index')
    fprintf('Selected features: %d\n', numel(Sol.index));
end

%% =========================
%  Optional: inspect solution
%  =========================
% Sol.w        : classifier coefficients
% Sol.b        : bias term
% Sol.nnz      : number of nonzero coefficients
% Sol.Iter     : number of IRL1 iterations
% Sol.TimeIter : CPU time per iteration
% Sol.index    : selected features (when available)