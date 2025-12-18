%% ============================================================
%      Cross Validation for Lp-XiSOCP SVM (Linear Kernel)
%
%% ============================================================

clear; clc; close all;

addpath(genpath('./dataset_bin_FS'));
addpath(genpath('./Models'));

%% === Load dataset ===
load('colorectal.mat');   datasetName='Colorectal';
% load('lymphoma_XY.mat'); datasetName='Lymphoma';
% load('pomeroy.mat');     datasetName='Pomeroy';

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
    fprintf("Using cvpartition (K-fold=%d).\n", CV);
    rng(1); 
    cv = cvpartition(Y,'KFold',CV);
    use_perm = false;
end

%% ============================================================
%            General parameters
%% ============================================================

FunPara.type = 'estim'; % 'estim' o 'chol'

p_values  = [0.1:0.1:0.9];
nu_values = [0.2, 0.4, 0.6, 0.8];
Cl = 5; Ch = 7;
ClCh=Cl:Ch;
C_values = 2.^ClCh;

np  = numel(p_values);
nNu = numel(nu_values);
nC  = numel(C_values);

%% === Preallocate ===
meanBACCU = zeros(np, nNu, nNu, nC);
meanACC   = zeros(np, nNu, nNu, nC);
meanF1    = zeros(np, nNu, nNu, nC);
stdBACCU  = zeros(np, nNu, nNu, nC);
stdACC    = zeros(np, nNu, nNu, nC);
stdF1     = zeros(np, nNu, nNu, nC);
meanNNZ   = zeros(np, nNu, nNu, nC);
meanTime  = zeros(np, nNu, nNu, nC);

%% ============================================================
%                   GRID SEARCH CV
%% ============================================================

fprintf('\n=== Cross-validation: exploring %d×%d×%d×%d models ===\n',...
    np,nNu,nNu,nC);

for ip = 1:np
    FunPara.p = p_values(ip);

for ic = 1:nC
    FunPara.c = C_values(ic);
    fprintf('\n==> Eval p=%.2f, C=2^{%d}\n', FunPara.p, Cl+ic-1);

for inu1 = 3:nNu
    nu1 = nu_values(inu1);

for inu2 = 1:nNu
    nu2 = nu_values(inu2);

    FunPara.kapa = sqrt([nu1, nu2] ./ (1 - [nu1, nu2]));

    fprintf('   → ν1=%.1f | ν2=%.1f\n', nu1, nu2);

    % CV containers
    BAC = zeros(1,CV);
    ACC = zeros(1,CV);
    F1s = zeros(1,CV);
    NNZ = zeros(1,CV);
    Time = zeros(1,CV);

    %% === FOLD LOOP ===
   for k = 1:CV

        if use_perm
            tst = perm(k:CV:m);
            trn = setdiff(1:m, tst);
        else
            trn = training(cv, k);
            tst = test(cv, k);
        end

        Xa = X(trn,:);     Ya = Y(trn);
        Xt = X(tst,:);     Yt = Y(tst);

        t0 = tic;
        [Pred, Sol] = Lp_XiSOCP_L1cvx(Xa, Ya, Xt, FunPara);
        Time(k) = toc(t0);

        [BAC(k), ACC(k), F1s(k)] = medi_auc_accu(Pred,Yt);
        NNZ(k) = Sol.nnz;
    end

     % === Store ===
    meanBACCU(ip, inu1, inu2, ic) = mean(BAC);
    meanACC(ip,  inu1, inu2, ic) = mean(ACC);
    meanF1(ip,   inu1, inu2, ic) = mean(F1s);

    stdBACCU(ip, inu1, inu2, ic) = std(BAC);
    stdACC(ip,   inu1, inu2, ic) = std(ACC);
    stdF1(ip,    inu1, inu2, ic) = std(F1s);

    meanNNZ(ip,  inu1, inu2, ic) = mean(NNZ);
    meanTime(ip, inu1, inu2, ic) = mean(Time);

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

[ipB, inu1B, inu2B, icB] = ind2sub(size(meanBACCU), idxB);
[ipA, inu1A, inu2A, icA] = ind2sub(size(meanACC), idxA);
[ipF, inu1F, inu2F, icF] = ind2sub(size(meanF1), idxF);

bestNNZ  = meanNNZ(idxB);
bestTime = meanTime(idxB);

fprintf('\n========= Best Results =========\n');

fprintf('→ Best BACCU: %.3f ± %.3f | p=%.2f | C=2^{%d} | ν1=%.1f | ν2=%.1f\n',...
    maxBACC, stdBACCU(idxB), p_values(ipB), ClCh(icB),...
    nu_values(inu1B), nu_values(inu2B));
fprintf('     Mean NNZ  = %.2f\n', bestNNZ);
fprintf('     Mean Time  = %.4f seg\n', bestTime);

fprintf('→ Best ACC  : %.3f ± %.3f | p=%.2f | C=2^{%d} | ν1=%.1f | ν2=%.1f\n',...
    maxACC, stdACC(idxA), p_values(ipA), ClCh(icA),...
    nu_values(inu1A), nu_values(inu2A));

fprintf('→ Best F1   : %.3f ± %.3f | p=%.2f | C=2^{%d} | ν1=%.1f | ν2=%.1f\n',...
    maxF1, stdF1(idxF), p_values(ipF), ClCh(icF),...
    nu_values(inu1F), nu_values(inu2F));

%% ============================================================
%     FLAT TABLE 
%% ============================================================

[Pg, Nu1g, Nu2g, Cg] = ndgrid(p_values, nu_values, nu_values, C_values);

T = table(...
    Pg(:), Nu1g(:), Nu2g(:), Cg(:), ...
    meanACC(:), stdACC(:), ...
    meanBACCU(:), stdBACCU(:), ...
    meanF1(:), stdF1(:), ...
    meanNNZ(:), meanTime(:), ...
    'VariableNames', {'p','nu1','nu2','C','ACC_mean','ACC_std','BACCU_mean','BACCU_std','F1_mean','F1_std','NNZ_mean','CPU_time'});

fprintf('\n--- First Row of the Results ---\n');
disp(T(1:min(10,height(T)),:))

%% ============================================================
%               Save of Results
%% ============================================================

timestamp = datestr(now,'yyyymmdd_HHMMSS');
file_name = sprintf('Results_LpXiSOCP_%s_%s.mat', datasetName, timestamp);

save(file_name, ...
    'datasetName', ...
    'p_values','nu_values','C_values', ...
    'meanACC','meanBACCU','meanF1','stdACC','stdBACCU','stdF1', ...
    'meanNNZ','meanTime','T', ...
    'maxACC','maxBACC','maxF1', ...
    'ipA','inu1A','inu2A','icA', ...
    'ipB','inu1B','inu2B','icB', ...
    'ipF','inu1F','inu2F','icF');

fprintf('\n=========================================================\n');
fprintf('   Results saved successfully.\n');
fprintf('   File: %s\n', file_name);
fprintf('=========================================================\n');
csv_file = sprintf('%s.csv', file_name);
writetable(T, csv_file);

fprintf('   Table T exported to  CSV: %s\n', csv_file);
fprintf('=========================================================\n');