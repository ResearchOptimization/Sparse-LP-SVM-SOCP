% =========================================================================
% medi_auc_accu
% =========================================================================
%
% Computes standard classification metrics for binary classification from
% predicted labels and true labels.
%
% This function returns:
%   - Balanced Accuracy
%   - Accuracy
%   - F1-score
%   - Sensitivity
%   - Specificity
%   - Precision
%   - Confusion matrix
%
% Inputs:
%   Predict : Predicted labels (vector)
%   Yt      : True labels (vector)
%
% Outputs:
%   Baccu : Balanced Accuracy
%   Accu  : Overall Accuracy
%   F1    : F1-score
%   Sens  : Sensitivity (recall for the positive class)
%   Spec  : Specificity (recall for the negative class)
%   Prec  : Precision (positive predictive value)
%   cm    : Confusion matrix [TN FP; FN TP]
%
% Notes:
%   - Labels are expected in {+1, -1} format.
%   - If 0 appears in Yt, it is automatically converted to -1.
%   - If Predict is identically zero, the function returns NaN values for
%     all metrics and for the confusion matrix.
% =========================================================================

function [Baccu,Accu,F1,Sens,Spec,Prec,cm]=medi_auc_accu(Predict,Yt)

% Ensure column vectors
Predict = Predict(:);
Yt = Yt(:);

% ---------------------------------------------------------
%%  Detect degenerate solution
% ---------------------------------------------------------
if all(Predict == 0)
    % No classifier exists → no metrics reported
    Baccu = NaN; Accu = NaN; F1 = NaN;
    Sens = NaN; Spec = NaN; Prec = NaN;
    cm = NaN(2,2);
    return
end

% Convert 0 labels to -1 if necessary
if any(Yt == 0)
    Yt(Yt == 0) = -1;
end

% Confusion matrix components
tPos = sum(Yt== 1 & Predict== 1);
tNeg = sum(Yt==-1 & Predict==-1);
fPos = sum(Yt==-1 & Predict== 1);
fNeg = sum(Yt== 1 & Predict==-1);

% Initialize outputs
Sens = NaN; Spec = NaN; Baccu = NaN; Accu = NaN; Prec = NaN; F1 = NaN;

% Accuracy
Accu = 1 - (sum(Predict ~= Yt) / length(Predict));

% Sensitivity & Specificity
if (tPos + fNeg) > 0
    Sens = tPos / (tPos + fNeg);
end

if (tNeg + fPos) > 0
    Spec = tNeg / (tNeg + fPos);
end

% Precision
if (tPos + fPos) > 0
    Prec = tPos / (tPos + fPos);
end

% F1-score
if ~isnan(Sens) && ~isnan(Prec) && (Sens + Prec) > 0
    F1 = 2 * (Prec * Sens) / (Prec + Sens);
end

% Balanced Accuracy
if ~isnan(Sens) && ~isnan(Spec)
    Baccu = 0.5 * (Sens + Spec);
end

% Confusion matrix [TN FP; FN TP]
cm = [tNeg, fPos; fNeg, tPos];

end

