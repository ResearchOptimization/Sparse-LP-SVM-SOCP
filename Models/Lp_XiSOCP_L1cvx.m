% =========================================================================
% Lp_XiSOCP_L1cvx
% =========================================================================
%
% This function implements the IRL1 algorithm (Algorithm 1 in the paper)
% for solving the ℓp-XiSOCP model with 0 < p < 1.
%
% At each iteration, the method solves a convex weighted ℓ1-SOCP subproblem
% using CVX, updates the reweighting coefficients, and checks convergence
% through the change in the weight vector.
%
% The optimization problem of interest is:
%
%   minimize   ||w||_p^p + c*Xi
%   subject to
%      kapa1 * ||S1' w|| <=  w' * mu1 + b - 1 + Xi
%      kapa2 * ||S2' w|| <= -w' * mu2 - b - 1 + Xi
%      Xi >= 0
%
% Inputs:
%   X        : Training data matrix (rows = samples, columns = features)
%   Y        : Training labels in {+1, -1}
%   Xt       : Test data matrix (rows = samples, columns = features)
%   FunPara  : Structure with fields:
%              - p     : ℓp quasi-norm exponent, with 0 < p < 1
%              - c     : Regularization parameter
%              - epsi  : Small positive smoothing parameter
%              - kapa  : Scalar or 2-vector [kapa1, kapa2]
%              - type  : Construction of S1 and S2 ('chol' or 'estim')
%
% Outputs:
%   Predict        : Predicted labels for Xt
%   Sol            : Structure containing solution details
%   Tf             : Total CPU time
%   Error_history  : History of the stopping error per iteration
%
% -------------------------------------------------------------------------
% Pseudocode: IRL1 Algorithm for the ℓp-XiSOCP Model
%
% Input:
%   Training data (X,Y), parameters (p, c, ε, κ1, κ2)
%
% Initialize:
%   k = 0
%   Φ^(0) = 1   (vector of ones)
%   w^(0) = initial vector
%
% Repeat:
%   1. Solve the convex weighted SOCP subproblem:
%
%        minimize   Σ_i Φ_i^(k) |w_i| + c * ξ
%        subject to
%           κ1 ||S1' w|| ≤  w'μ1 + b - 1 + ξ
%           κ2 ||S2' w|| ≤ -w'μ2 - b - 1 + ξ
%           ξ ≥ 0
%
%   2. Update the reweighting coefficients:
%
%        Φ_i^(k+1) = p / (|w_i^(k)| + ε)^(1-p)
%
%   3. Check convergence:
%
%        if ||w^(k+1) - w^(k)||_∞ < Tol
%           stop
%
%   4. Set k = k + 1
%
% Output:
%   Final solution (w, b)
%   Selected features obtained from the final sparse solution
% -------------------------------------------------------------------------
%
% Construction of matrices S1 and S2:
%
% The matrices S_i are defined so that
%
%       S_i S_i' ≈ Σ_i,
%
% where Σ_i denotes the covariance matrix of class i.
%
% Two options are available:
%
%   1) 'chol'
%      S_i is obtained via Cholesky factorization of Σ_i
%      S_i ∈ R^{n×n}
%
%   2) 'estim'
%      S_i is computed directly from the data as
%      S_i = (X_i - μ_i e^T)/sqrt(m_i)
%      S_i ∈ R^{n×m_i}
%
% Note:
%   Both constructions yield the same quadratic form:
%
%       ||S_i' w||_2^2 = w' Σ_i w
%
%   The 'estim' option avoids explicit covariance computation and is more
%   suitable for high-dimensional settings (n >> m).
% =========================================================================

function [Predict, Sol, Tf,Error_history]=Lp_XiSOCP_L1cvx(X, Y, Xt, FunPara)
% ---------- Input checks and defaults ----------
if nargin < 4, 
    error('Requires X, Y, Xt and FunPara');
end
[nSamples, n] = size(X);

% defaults
if ~isfield(FunPara,'p'), FunPara.p = 0.5; end
if ~isfield(FunPara,'c'), FunPara.c = 1; end
if ~isfield(FunPara,'epsi'), FunPara.epsi = 1e-6; end
if ~isfield(FunPara,'kapa'), 
    alpha=[0.2;0.2];
    FunPara.kapa = sqrt(alpha./(1-alpha)); end
if ~isfield(FunPara,'type'), FunPara.type = 'chol'; end

Tinic=cputime;

p = FunPara.p;
c = FunPara.c;
epsi = FunPara.epsi;
kapa = FunPara.kapa;
Factor_type = FunPara.type;

% ensure kapa is 2-element vector
if isscalar(kapa)
    kapa = [kapa, kapa];
elseif numel(kapa) >= 2
    kapa = kapa(1:2);
else
    error('kapa must be scalar or vector with at least 2 elements');
end

% ---------- Split classes ----------
fin1 = (Y==1);
fin2 = (Y==-1);
A = X(fin1,:);
B = X(fin2,:);
if isempty(A) || isempty(B)
    error('Both classes must have at least one example.');
end

% ---------- Statistical measures ----------
mu(1,:) = mean(A,1);
mu(2,:) = mean(B,1);

switch lower(Factor_type)
    case 'chol'
        % covariance + small ridge for numeric stability
        Sigma(:,:,1) = cov(A) + 1e-7*eye(n);
        Sigma(:,:,2) = cov(B) + 1e-7*eye(n);
        S1 = chol(Sigma(:,:,1),'lower'); % n x n
        S2 = chol(Sigma(:,:,2),'lower');
    case 'estim'
        Num_c1 = size(A,1);
        Num_c2 = size(B,1);
        % estimates of S = (T - mu*e')/sqrt(N)
        S1 = (A' - mu(1,:)'*ones(1,Num_c1))/sqrt(max(1,Num_c1));
        S2 = (B' - mu(2,:)'*ones(1,Num_c2))/sqrt(max(1,Num_c2));
    otherwise
        error('Unknown FunPara.type: use ''chol'' or ''estim''.');
end

% -------------------------------------------------------------------------
% Algorithm 1 (IRL1): Iteratively Reweighted ℓ1 scheme
% -------------------------------------------------------------------------
N_iter = 50;
Phi = ones(n,1);     % initial weights (column)
wn = 10*ones(n,1);
Tol = 1e-3;

% Preallocate histories
W_history = zeros(n, N_iter);
Val_obj = nan(1,N_iter);
TimeIter = nan(1,N_iter);
Error_history = nan(1,N_iter);
nnzs = nan(1,N_iter);

for k = 1:N_iter
    tstart = tic;
    % Step 1: Solve weighted convex SOCP subproblem (CVX)
    try
        cvx_begin quiet
            % if you want to use a specific solver, set it externally
            cvx_solver sedumi
            cvx_precision('medium')
            variables w(n) b Xi
            minimize( sum(Phi .* abs(w)) + c*Xi )
            subject to
                kapa(1)*norm(S1' * w, 2)<=w' * mu(1,:)'+b-1+Xi;
                kapa(2)*norm(S2' * w, 2)<=-w' * mu(2,:)'-b-1+Xi;
                Xi>=0;
        cvx_end
    catch ME
        warning('CVX solver failed on iteration %d: %s', k, ME.message);
        % return last known solution if available
        break;
    end

    % metrics
    nnz_curr = sum(abs(w) > 1e-5);
    nnzs(k) = nnz_curr;
    Val_obj(k) = cvx_optval;
    TimeIter(k) = toc(tstart);
    W_history(:,k) = w; % store raw w (useful for debugging)

    if nnz_curr == 0
        % trivial solution, set outputs and exit loop
        Sol.iter = k;
        w = zeros(n,1);
        b = 0;
        break
    end

    % Step 2: Update IRL1 weights
    % Φ_i = p / (|w_i| + ε)^(1-p)
    % Phin corresponds to: p/(|w|+eps)^(1-p)

    Phin=p./((abs(w) + epsi).^(1-p));
    Error_history(k) = norm(wn-w, Inf);

    % Step 3: Check convergence (stopping criterion)
    if Error_history(k) < Tol
        Phi = Phin;
        wn = w;
        % store final Phin in history and break
        W_history(:,k) = Phin;
        break;
    else
        Phi = Phin;
        wn = w;
    end
end
Tf=cputime-Tinic;

% trim history arrays to actual iterations
last_iter = find(~isnan(Val_obj),1,'last');
if isempty(last_iter)
    last_iter = 0;
end
Val_obj = Val_obj(1:last_iter);
TimeIter = TimeIter(1:last_iter);
W_history = W_history(:,1:last_iter);
nnzs = nnzs(1:last_iter);
Error_history = Error_history(1:last_iter);

% ---------- Postprocessing ----------
Sol.w = w;
Sol.b = b;
Sol.nnzs = nnzs;
Sol.ValObj = Val_obj;
Sol.Phi = W_history;
Sol.TimeIter = TimeIter;
Sol.Iter = last_iter;

if sum(abs(w) > 1e-6) > 0
    % compute lower bound L per selected feature (only if denominator positive)
    L = zeros(n,1);
    for i = 1:n
        den=c*abs(mu(1,i)-mu(2,i))+ kapa(1)*norm(S1(i,:))+kapa(2)*norm(S2(i,:));
        if den > 0
            L(i) = (2*p / den)^(1/(1-p));
        else
            L(i) = 0;
        end
    end
    Sol.L = L;
    Index=find(abs(w)>=L); 
    Sol.index=Index;
    % predicted labels for Xt using selected features
    Sol.Predict_fs = sign(Xt(:,Index) * w(Index) + b); %
else 
    Sol.Predict_fs=zeros(size(Xt,1),1);
end

% final outputs
Val_Xt = Xt * w + b;
Sol.val = Val_Xt;
Predict = Sol.Predict_fs;% sign(Val_Xt);
Sol.nnz = sum(abs(w) > 1e-5);
Sol.Ind=find(abs(w)>1e-5);
end
