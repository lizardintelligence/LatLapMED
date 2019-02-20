function [y_hat, model] = LatLapMED(X, y, G_args, L_args, K_args, phi, B, C, use_pdist)

% Requires CVX or LibSVM called in Solve_Convex()
%
% Input arguments:
% X - data points
% y - high utility = 1, low utility = -1, unlabeled = 0
% G_args - distance metric and GEM arguments
% L_args - distance metric and Laplacian parameters
% K_args - kernel type and parameters
% phi - percentage of anomalous points
% beta - smoothness parameter
% C - cost
% use_pdist - if true, use Matlab's pdist2 function
%
% Output arguments:
% y_hat - predicted labels
% model - structure that holds all training parameters

[n, p] = size(X);
l_idx = find(y~=0); l = size(l_idx, 1); % index of label pts
y_hat = -1*ones(n, 1);

EM_max_val = [];
model = {};
iter = 0;
diff = 10^8;
while diff > 0 && iter < 500

    old_y_hat = y_hat;
    
    % E-Step: Use empirical dist. of non-high utility pts. to estimate etas
    if iter == 0 % no predicted distance from margin
        [eta_hat, s_dist] = GEM(X, y, zeros(n, 1), phi, G_args, {}, use_pdist);
        a = sum(eta_hat, 1);
        B = (n^2/sum(eta_hat, 1)^2) * B; % if fewer than n points, normalize
    else
        [eta_hat, ~] = GEM(X, y, d_hat, phi, G_args, s_dist, use_pdist);
        
    end

    % M-Step: Maximize alpha
    L = CreateLaplacian(X(eta_hat~=0, :), L_args, use_pdist);    
    K = KernalFunc(X(eta_hat~=0, :), X(eta_hat~=0, :), K_args, use_pdist);
    if B == 0, mod_K = K; else, mod_K = K/(eye(a) + 2*B*L*K); end

    [alpha_hat, bias_hat, opt_max] = SolveConvex(y, eta_hat, mod_K, C, 'med'); 
    if iter > 0 && opt_max < EM_max_val(end) % EM is not increasing anymore so stop
        break
    end  
    
    % Prediction
    d_hat = -10^10*ones(n, 1); % nominal points are all very far from margin
    d_hat(eta_hat~=0) = mod_K * (y(eta_hat~=0) .* alpha_hat) + bias_hat; 
    y_hat = sign(d_hat); 
    y_hat(l_idx) = y(l_idx); % labelled points are predicted as such
    k_func = @(xx) KernalFunc(xx, X(eta_hat~=0, :), K_args, use_pdist) ...
                               / (eye(size(K, 1)) + 2*B*L*K); 
    g_func = @(xx, dd) GEM([X; xx], [y; zeros(size(xx,1),1)], [d_hat; dd], ...
                            phi, G_args, {}, use_pdist);
    pred_func = @(k) k * (y(eta_hat~=0) .* alpha_hat) + bias_hat; 
                
    EM_max_val = [EM_max_val, opt_max]; % hold maximum value
    diff = sum(abs(y_hat - old_y_hat))/2; % loop until distributions is stable 
    iter = iter + 1;

    % Store Model Information
    model.y = y_hat;
    model.eta = eta_hat;
    model.alpha = alpha_hat;
    model.bias = bias_hat;
    model.margin_dist = d_hat;
    model.phi = phi;
    model.beta = B;
    model.cost = C;
    model.g_args = G_args;
    model.l_args = L_args;
    model.k_args = K_args;
    model.L = L;
    model.K = mod_K;
    model.g_func = g_func;
    model.k_func = k_func;
    model.pred_func = pred_func;
    model.max = EM_max_val;
    model.iter = iter;
end

end