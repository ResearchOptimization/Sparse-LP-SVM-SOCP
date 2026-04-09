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
%     Lp_XiSOCP_L1cvx.m  : Main optimization routine implementing an
%                         Iteratively Reweighted L1 (IRL1) algorithm
%                         solved via CVX for the Lp-SOCP model (0 < p < 1).
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
%    - Each dataset must provide:
%      X : data matrix (rows = samples, columns = features)
%      Y : label vector with entries in {+1, -1}
%
% 2. Cross-Validation Program:
%    - CV_Lp_XiSOCP.m :
%      Performs grid-search cross-validation over (p, ν1, ν2, C)
%      for the Lp-XiSOCP model with linear decision function.
%
% 3. Evaluation Metrics:
%    - medi_auc_accu.m :
%      Computes classification metrics such as Accuracy, Balanced Accuracy
%      (BACCU), F1-score, Sensitivity, Specificity, Precision, and the
%      confusion matrix.
%
% Requirements:
% -------------
% The system requirements for running this code are:
%
% - MATLAB (tested on versions R2022a, R2022b, R2023, and R2024)
% - CVX toolbox (required to solve SOCP subproblems)
%   * Download from: http://cvxr.com/cvx/
%   * After installation, run: cvx_setup
%
% - Statistics and Machine Learning Toolbox (optional)
%   * Required only if cvpartition is used for cross-validation.
%   * Not needed if a predefined partition vector "perm" is provided.
%
% Note:
%   - The current implementation calls CVX with SeDuMi as solver inside
%     Lp_XiSOCP_L1cvx.m.
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
%         - meanTime   : Average CPU time per fold
%
%         Results are automatically saved in:
%         - .mat file (full workspace results)
%         - .csv file (flattened summary table)
%
% Parameters:
% -----------
% The main parameters are defined in CV_Lp_XiSOCP.m and passed through the
% structure FunPara:
%
% - p      : ℓp quasi-norm exponent (0 < p < 1)
% - c      : regularization parameter
% - epsi   : small positive constant for numerical stability in IRL1
% - kapa   : vector [κ1, κ2] controlling the SOCP constraints,
%            typically derived from the confidence parameters (ν1, ν2)
%
% - type   : method to construct matrices S1 and S2:
%
%            'chol'  : uses Cholesky factorization of covariance matrices
%                      (S_i ∈ ℝ^{n×n})
%
%            'estim' : uses sample-based estimation
%                      S_i = (X_i - μ_i e^T)/sqrt(m_i)
%                      (S_i ∈ ℝ^{n×m_i})
%
% Recommendation:
% - Use 'estim' for high-dimensional datasets (n >> m), e.g., microarrays
% - Use 'chol' when covariance matrices are well-conditioned (m >> n)
%
% Basic Example:
% --------------
% The file example_LpXiSOCP.m provides a minimal working example of how to
% use the function Lp_XiSOCP_L1cvx.m without cross-validation.
%
% It includes:
%   - Loading a dataset
%   - Setting model parameters (p, c, κ, type)
%   - Training the model
%   - Computing performance metrics
%
% A simplified version of the workflow is:
%
%   load('colorectal.mat');
%
%   FunPara.type = 'estim';      % 'estim' or 'chol'
%   FunPara.p    = 0.7;
%   FunPara.c    = 2^6;
%   nu           = [0.2; 0.8];
%   FunPara.kapa = sqrt(nu./(1 - nu));     
%
%   [Prediction, Sol] = Lp_XiSOCP_L1cvx(X, Y, X, FunPara);
%
%   [BACCU, ACCU, F1] = medi_auc_accu(Prediction, Y);
%
% This example is intended as an entry point to understand how the IRL1-based
% ℓp-SOCP model is used in practice before running full cross-validation.
%
% Examples:
% ---------
% Example 1: Run the basic example
%   - Run example_LpXiSOCP.m
%
% Example 2: Run CV of Lp–XiSOCP SVM on 'colorectal.mat'
%   - Uncomment dataset loading line in CV_Lp_XiSOCP.m
%   - Run the script
%
% Example 3: Run CV of Lp–XiSOCP SVM on 'pomeroy.mat'
%   - Change dataset selection
%   - Run the script
%
% Contact:
% --------
% - Miguel Carrasco   : macarrasco@miuandes.cl
% - Benjamin Ivorra   : ivorra@ucm.es
% - Julio López       : julio.lopez@udp.cl
