function [Predict, Sol, Tf,Error_history]=Lp_XiSOCP_L1cvx(X, Y, Xt, FunPara)
% Reweighted L1 Algorithm for solving the Lp-SOCP by CVX solver
%   min |w|_p^p + c*Xi  
%   s.t. kapa1 * ||S1' w|| <= w' mu1 + b - 1 + Xi
%        kapa2 * ||S2' w|| <= -w' mu2 - b - 1 + Xi
%        Xi >= 0
%
% Inputs:
%   X, Y      - training data (rows = samples, cols = features), labels in {1,-1}
%   Xt        - test data (rows = samples, cols = features)
%   FunPara   - struct with fields:
%                p (0<p<1), c, epsi (small positive), kapa (scalar or 2-vector),
%                type ('chol' or 'estim')
%
% Outputs:
%   Predict   - predicted labels for Xt (sign of score)
%   Sol       - struct with solution details (w,b,indices,nnzs,ValObj,...)
%   Error_history - history of weight-change error per iteration


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
    nu=[0.2;0.2];
    FunPara.kapa = sqrt(nu./(1-nu)); end
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
        Mchol_1 = chol(Sigma(:,:,1),'lower'); % n x n
        Mchol_2 = chol(Sigma(:,:,2),'lower');
    case 'estim'
        Num_c1 = size(A,1);
        Num_c2 = size(B,1);
        % estimates of S = (T - mu*e')/sqrt(N)
        Mchol_1 = (A' - mu(1,:)'*ones(1,Num_c1))/sqrt(max(1,Num_c1));
        Mchol_2 = (B' - mu(2,:)'*ones(1,Num_c2))/sqrt(max(1,Num_c2));
    otherwise
        error('Unknown FunPara.type: use ''chol'' or ''estim''.');
end

% ---------- Iterative reweighted L1 ----------
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
    try
        cvx_begin quiet
            % if you want to use a specific solver, set it externally
            cvx_solver sedumi
            cvx_precision('medium')
            variables w(n) b Xi
            minimize( sum(Phi .* abs(w)) + c*Xi )
            subject to
                kapa(1)*norm(Mchol_1' * w, 2)<=w' * mu(1,:)'+b-1+Xi;
                kapa(2)*norm(Mchol_2' * w, 2)<=-w' * mu(2,:)'-b-1+Xi;
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

    % update reweighting
    % Phin corresponds to: p/(|w|+eps)^(1-p)
    Phin=p./((abs(w) + epsi).^(1-p));
    Error_history(k) = norm(wn-w, Inf);

    % convergence check
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
        den=c*abs(mu(1,i)-mu(2,i))+ kapa(1)*norm(Mchol_1(i,:))+kapa(2)*norm(Mchol_2(i,:));
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

end
