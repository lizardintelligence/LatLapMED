clc; clear;

trials = 5;
seed = 94454;
Phi_seq = [.02:.01:.1, .15:.1:.55, .75, 1];
fast = true; % uses the pdist2 function in Matlab Stat and ML Toolbox

hold_all_LatLapMED_rates = cell(trials, 1);
hold_all_RF_rates = cell(trials, 1);
% parfor t = 1:trials
for t = 1:trials
   
    fprintf('trial: %d \n', t);
    
    rates_LatLapMED = Inf*ones(1, size(Phi_seq, 2), 4);
    rates_RF = Inf*ones(1, size(Phi_seq, 2), 4);

    % Generate Data
    [X, eta, utility, y_obvs] = GenerateData(5000, 2, 0.3, 0.05, 'folded t', rng(t+seed), 'shuffle');
%     [X, eta, utility, y_obvs] = GenerateData(365*10, 30, 0.15, 0.1, 'poisson', rng(t+seed), 'shuffle');
    [n, p] = size(X); l_idx = find(y_obvs~=0); l = size(l_idx, 1); 
    X = (X - repmat(mean(X, 1), n, 1)) ./ repmat(std(X, 1), n, 1); % normalize X
    
    % Parameters
    GEM_args = 'euclid 10 1'; % number of neighbors and threshold of how far from margin is significant
    Lap_args = 'euclid 50 100'; % type, number of neighbors, heat kernel width for graph Laplacian
    Kern_args = 'rbf 0 0 0 1'; % type, scale, intercept, power, rbf_sig
    Phi = sum(eta)/n; % percent of data assumed to be anomalous
    Cost = 10; 
    gamma_A = 1 / 2*Cost*l; % penalty_obvs for ambient space (RKHS)
    gamma_I = 5; % penalty_obvs for intrinisic geometry_obvs (data distribution manifold)
    Beta = (gamma_I*Cost*l / n^2); 

    for phi = 1:size(Phi_seq, 2)
        
        % Train
        [y_hat, ~] = LatLapMED(X, y_obvs, GEM_args, Lap_args, Kern_args, Phi_seq(phi), Beta, Cost, fast);
        [rates_LatLapMED(1, phi, :), ~] = GetRates(utility, y_hat, 'LatLapMED', false);

        % 1-class SVM + Random Forest (requires LibSVM and Matlab Stat and ML Toolbox)
        nu_svm_model = svmtrain(ones(n, 1), X, sprintf('-q, -s 2, -t 2, -g .5, -n %f', Phi_seq(phi))); 
        eta_1svm = abs(y_obvs); eta_1svm(nu_svm_model.sv_indices) = 1;
        RF_model = TreeBagger(100, X(l_idx, :), y_obvs(l_idx));
        y_hat_RF = -1*ones(n, 1); 
        y_hat_RF(eta_1svm~=0) = str2double(predict(RF_model, X(eta_1svm~=0, :)));
        y_hat_RF(l_idx) = y_obvs(l_idx); % labelled points are predicted as such
        y_hat_RF = sign(y_hat_RF); % convert to signs    
        [rates_RF(1, phi, :), ~] = GetRates(utility, y_hat_RF, 'nuSVM + RF', false);        

    end
    
    hold_all_LatLapMED_rates{t} = rates_LatLapMED;
    hold_all_RF_rates{t} = rates_RF;
end

mean_LatLapMED_rates = mean(cat(4, hold_all_LatLapMED_rates{:}), 4);
mean_RF_rates = mean(cat(4, hold_all_RF_rates{:}), 4);

% Plot ROC curves
Figure1=figure(1); clf(Figure1); 
set(Figure1, 'Color', 'w', 'Units', 'inches', 'Position', [0,0, 7, 5]);
plot([mean_LatLapMED_rates(1, :, 1), 1], 1-[mean_LatLapMED_rates(1, :, 2), 0], ...
    '-', 'LineWidth', 1.5, 'MarkerSize', .5);
hold on
plot([mean_RF_rates(1, :, 1), 1], 1-[mean_RF_rates(1, :, 2), 0], ...
    '-', 'LineWidth', 1.5, 'MarkerSize', .5);
set(gca,'box', 'off')
xlabel('FPR', 'FontSize', 10); ylabel('TPR', 'FontSize', 10); 
hold off;
set(gca, 'LooseInset',get(gca,'TightInset')) % fix margins

