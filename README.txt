% =============================================================================
% Title: Sparse Feature Selection via ℓp-Quasi-Norm Second-Order Cone Programming
% =============================================================================
%
% Overview:
% This repository provides MATLAB code to run cross-validation (CV) for binary
% classification using an ℓp-quasi-norm (0 < p < 1) combined with a
% Second-Order Cone Programming (SOCP) formulation (linear kernel).
%
% The method promotes sparse feature selection while maintaining robustness
% through second-order cone constraints.
%
% Folder Structure:
% -----------------
% Models/                
%     Lp_XiSOCP_L1cvx.m  : Main optimization routine (CVX-based).
%
% dataset_bin_FS/        
%     colorectal.mat
%     lymphoma_XY.mat
%     pomeroy.mat
%
% Main Components:
% ----------------
% 1. Datasets (dataset_bin_FS/):
%    - Benchmark binary classification datasets in .mat format.
%
% 2. Cross-Validation Program:
%    - CV_Lp_XiSOCP.m :
%      Performs grid-search cross-validation over (p, ν1, ν2, C)
%      for the Lp–XiSOCP SVM model (linear kernel).
%
% 3. Evaluation Metrics:
%    - medi_auc_accu.m :
%      Computes Accuracy, Balanced Accuracy (BACCU), F1-score, etc.
%
% Requirements:
% -------------
% - MATLAB R2021b or later
% - CVX (required for solving SOCP subproblems)
% - Statistics and Machine Learning Toolbox
%   (used for cvpartition; optional if a permutation vector "perm" is provided)
%
% Usage:
% ------
% Step 1: Configure parameters in CV_Lp_XiSOCP.m
%         - Dataset selection (e.g., 'colorectal.mat')
%         - Number of CV folds (default: 10)
%
% Step 2: Run the script in MATLAB:
%         - Loads data
%         - Performs grid-search CV
%         - Evaluates performance metrics
%
% Step 3: Inspect results:
%         - meanACC    : Mean accuracy over folds
%         - meanBACCU  : Mean balanced accuracy
%         - meanF1     : Mean F1-score
%         - meanNNZ    : Average number of selected features
%         - meanTime  : Average CPU time per fold
%
%         Results are automatically saved in:
%         - .mat file (full workspace results)
%         - .csv file (flattened summary table)
%
% Examples:
% ---------
% Example 1: Run Lp–XiSOCP SVM on 'colorectal.mat'
%   - Uncomment dataset loading line in CV_Lp_XiSOCP.m
%   - Run the script
%
% Example 2: Run Lp–XiSOCP SVM on 'pomeroy.mat'
%   - Change dataset selection
%   - Run the script
%
% Contact:
% --------
% - Miguel Carrasco   : macarrasco@miuandes.cl
% - Benjamin Ivorra   : ivorra@ucm.es
% - Julio López       : julio.lopez@udp.cl
