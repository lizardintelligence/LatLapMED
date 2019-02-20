function [Rates, Raw] = GetRates(util, y_hat, model_name, print)

% Input arguments:
% util - utility = 1, no utility = 0
% y_hat - model's predicted labels
% model_name - which model
% print - display rates
%
% Output arguments:
% Rates - FPR, FNR, Recall, Precision
% Raw - FP, FN, TP, TN

FP = sum(1 - util(y_hat == 1)); % num predicted as important anomalies but have 0 utility
TP = sum(util(y_hat == 1));  % num correctly predicted as utility 1 (subtract num visible ones)
FN = sum(util(y_hat <= 0));  % num predicted as no utilty, actually have utility 1
TN = sum(1 - util(y_hat <= 0)); % num correctly predicted as utility 0

FPR = FP / (FP + TN); % Type I error
FNR = FN / (FN + TP); % Type II error
Recall = 1 - FNR;
Precision = TP / (TP + FP); 

% Print Output
if print == true
    disp(' ');
    disp(model_name);
    disp(strcat('FPR:  ', num2str(FPR), ' FNR:  ', num2str(FNR), ...
        ' Recall:  ', num2str(Recall), ' Precision:  ', num2str(Precision)));
    disp(' ');
end

Rates = [FPR, FNR, Recall, Precision];
Raw = [FP, FN, TP, TN];

end

