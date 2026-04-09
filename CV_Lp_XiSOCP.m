% =========================================================================
% Cross-Validation for Lp-XiSOCP SVM (Linear Kernel)
% =========================================================================
%
% This script performs grid-search cross-validation for the Lp-XiSOCP SVM
% model with linear kernel.
%
% The hyperparameters explored are:
%   - p       : ℓp quasi-norm exponent
%   - alpha1  : confidence parameter for class +1
%   - alpha2  : confidence parameter for class -1
%   - C       : regularization parameter
%
% For each parameter combination, the script:
%   1. Splits the data into training and test folds,
%   2. Trains the model using Lp_XiSOCP_L1cvx.m,
%   3. Evaluates the classifier using BACCU, ACC, F1, NNZ, and CPU time,
%   4. Stores the mean and standard deviation of the results across folds.
%
% The script also identifies the best parameter configuration according to
% the average Balanced Accuracy and saves the full results in MAT and CSV
% formats.
% =========================================================================

clear; clc; close all;

addpath(genpath('./dataset_bin_FS'));
addpath(genpath('./Models'));

%% === Load dataset ===
load('colorectal.mat');   datasetName='Colorectal';
% load('lymphoma_XY.mat'); datasetName='Lymphoma';
% load('pomeroy.mat');     datasetName='Pomeroy';
% load('DermaMNIST_All_0vs2.mat'); datasetName='DermaMNIST0vs2';

[m, n] = size(X);
fprintf('Load Dataset: %d Samples, %d Features.\n', m, n);

%% ============================================================
%     Partition handling (perm o CV)
%% ============================================================

CV = 10;  % default

if exist('perm','var') && numel(perm)==m
    fprintf('\nUsing provided partition "perm".\n');
    use_perm = true;
else
    fprintf('Using cvpartition (K-fold=%d).\n', CV);
    rng(1);
    cv = cvpartition(Y,'KFold',CV);
    use_perm = false;
end

%% ============================================================
%            General parameters
%% ============================================================

FunPara.type = 'estim'; % 'estim' or 'chol'

p_values      = 0.1:0.1:0.9;
alpha_values  = [0.2, 0.4, 0.6, 0.8];

Cl = 5; Ch = 7;
ClCh    = Cl:Ch;          % exponent grid
C_values = 2.^ClCh;       % actual C values

np  = numel(p_values);
nAl = numel(alpha_values);
nC  = numel(C_values);

%% === Preallocate ===
meanBACCU = zeros(np, nAl, nAl, nC);
meanACC   = zeros(np, nAl, nAl, nC);
meanF1    = zeros(np, nAl, nAl, nC);

stdBACCU  = zeros(np, nAl, nAl, nC);
stdACC    = zeros(np, nAl, nAl, nC);
stdF1     = zeros(np, nAl, nAl, nC);

meanNNZ   = zeros(np, nAl, nAl, nC);
meanTime  = zeros(np, nAl, nAl, nC);

%% ============================================================
%                   GRID SEARCH CV
%% ============================================================

fprintf('\n=== Cross-validation: exploring %d×%d×%d×%d models ===\n', np, nAl, nAl, nC);

for ip = 1:np
    FunPara.p = p_values(ip);

    for ic = 1:nC
        FunPara.c = C_values(ic);
        fprintf('\n==> Eval p=%.2f, C=2^{%d}\n', FunPara.p, ClCh(ic));

        for ial1 = 1:nAl
            alpha1 = alpha_values(ial1);

            for ial2 = 1:nAl
                alpha2 = alpha_values(ial2);
                
                FunPara.kapa = sqrt([alpha1, alpha2] ./ (1 - [alpha1, alpha2]));

                fprintf('   → alpha1=%.1f | alpha2=%.1f\n', alpha1, alpha2);

                % CV containers
                BAC  = zeros(1, CV);
                ACC  = zeros(1, CV);
                F1s  = zeros(1, CV);
                NNZ  = zeros(1, CV);
                Time = zeros(1, CV);

                %% === FOLD LOOP ===
                for k = 1:CV

                    if use_perm
                        tst = perm(k:CV:m);
                        trn = setdiff(1:m, tst);
                    else
                        trn = training(cv, k);   % logical index
                        tst = test(cv, k);       % logical index
                    end

                    Xa = X(trn,:);     Ya = Y(trn);
                    Xt = X(tst,:);     Yt = Y(tst);

                    t0 = tic;
                    [Pred, Sol] = Lp_XiSOCP_L1cvx(Xa, Ya, Xt, FunPara);
                    Time(k) = toc(t0);

                    [BAC(k), ACC(k), F1s(k)] = medi_auc_accu(Pred, Yt);
                    NNZ(k) = Sol.nnz;
                end

                % === Store ===
                meanBACCU(ip, ial1, ial2, ic) = mean(BAC);
                meanACC(ip,   ial1, ial2, ic) = mean(ACC);
                meanF1(ip,    ial1, ial2, ic) = mean(F1s);

                stdBACCU(ip, ial1, ial2, ic) = std(BAC);
                stdACC(ip,   ial1, ial2, ic) = std(ACC);
                stdF1(ip,    ial1, ial2, ic) = std(F1s);

                meanNNZ(ip,  ial1, ial2, ic) = mean(NNZ);
                meanTime(ip, ial1, ial2, ic) = mean(Time);

            end
        end
    end
end

%% ============================================================
%          Best Model Selection
%% ============================================================

[maxBACC, idxB] = max(meanBACCU(:));
[maxACC,  idxA] = max(meanACC(:));
[maxF1,   idxF] = max(meanF1(:));

[ipB, ial1B, ial2B, icB] = ind2sub(size(meanBACCU), idxB);
[ipA, ial1A, ial2A, icA] = ind2sub(size(meanACC),   idxA);
[ipF, ial1F, ial2F, icF] = ind2sub(size(meanF1),    idxF);

bestNNZ  = meanNNZ(idxB);
bestTime = meanTime(idxB);

fprintf('\n========= Best Results =========\n');

fprintf('→ Best BACCU: %.3f ± %.3f | p=%.2f | C=2^{%d} | alpha1=%.1f | alpha2=%.1f\n', ...
    maxBACC, stdBACCU(idxB), p_values(ipB), ClCh(icB), ...
    alpha_values(ial1B), alpha_values(ial2B));
fprintf('     Mean NNZ   = %.2f\n', bestNNZ);
fprintf('     Mean Time  = %.4f seg\n', bestTime);

fprintf('→ Best ACC  : %.3f ± %.3f | p=%.2f | C=2^{%d} | alpha1=%.1f | alpha2=%.1f\n', ...
    maxACC, stdACC(idxA), p_values(ipA), ClCh(icA), ...
    alpha_values(ial1A), alpha_values(ial2A));

fprintf('→ Best F1   : %.3f ± %.3f | p=%.2f | C=2^{%d} | alpha1=%.1f | alpha2=%.1f\n', ...
    maxF1, stdF1(idxF), p_values(ipF), ClCh(icF), ...
    alpha_values(ial1F), alpha_values(ial2F));

%% ============================================================
%     FLAT TABLE
%% ============================================================

[Pg, A1g, A2g, Cg] = ndgrid(p_values, alpha_values, alpha_values, C_values);

T = table(...
    Pg(:), A1g(:), A2g(:), Cg(:), ...
    meanACC(:), stdACC(:), ...
    meanBACCU(:), stdBACCU(:), ...
    meanF1(:), stdF1(:), ...
    meanNNZ(:), meanTime(:), ...
    'VariableNames', {'p','alpha1','alpha2','C','ACC_mean','ACC_std','BACCU_mean','BACCU_std','F1_mean','F1_std','NNZ_mean','CPU_time'});

fprintf('\n--- First Rows of the Results ---\n');
disp(T(1:min(10,height(T)),:))

%% ============================================================
%               Save of Results
%% ============================================================

timestamp = datestr(now,'yyyymmdd_HHMMSS');
base_name = sprintf('Results_LpXiSOCP_%s_%s', datasetName, timestamp);

mat_file = sprintf('%s.mat', base_name);
csv_file = sprintf('%s.csv', base_name);

save(mat_file, ...
    'datasetName', ...
    'p_values','alpha_values','C_values', ...
    'meanACC','meanBACCU','meanF1','stdACC','stdBACCU','stdF1', ...
    'meanNNZ','meanTime','T', ...
    'maxACC','maxBACC','maxF1', ...
    'ipA','ial1A','ial2A','icA', ...
    'ipB','ial1B','ial2B','icB', ...
    'ipF','ial1F','ial2F','icF');

writetable(T, csv_file);

fprintf('\n=========================================================\n');
fprintf('   Results saved successfully.\n');
fprintf('   MAT: %s\n', mat_file);
fprintf('   CSV: %s\n', csv_file);
fprintf('=========================================================\n');
