function y_new = PredictLatLapMED(X_new, model)

% Input arguments:
% X_new - new data points
% model - trained model
% 
% Output arguments:
% y_new - predicted labels of new data points

mod_K = model.k_func(X_new);
d_new = model.pred_func(mod_K);
[eta_total, ~] = model.g_func(X_new, d_new);
eta_new = eta_total((size(eta_total, 1)-size(X_new, 1)+1):end);
y_new = sign(d_new); 
y_new(eta_new == 0) = -1; % nominal points = low utility

end