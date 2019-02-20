clc; clear;

% Generate Data
[X, eta, utility, y_obvs] = GenerateData(5000, 2, 0.3, 0.05, 'folded t', 'shuffle', 'shuffle');
% [X, eta, utility, y_obvs] = GenerateData(365*10, 30, 0.15, 0.1, 'poisson', 'shuffle', 'shuffle');
[n, p] = size(X); l_idx = find(y_obvs~=0); l = size(l_idx, 1); 
X = (X - repmat(mean(X, 1), n, 1)) ./ repmat(std(X, 1), n, 1); % normalize X

% Parameters
GEM_args = 'euclid 10 1'; % number of neighbors and threshold of how far from margin is significant
Lap_args = 'euclid 50 100'; % type, number of neighbors, heat kernel width for graph Laplacian
Kern_args = 'rbf 0 0 0 1'; % type, scale, intercept, power, rbf_sig
Phi = sum(eta)/n; % percent of data assumed to be anomalous
Cost = 150; 
gamma_A = 1 / 2*Cost*l; % penalty_obvs for ambient space (RKHS)
gamma_I = 10; % penalty_obvs for intrinisic geometry_obvs (data distribution manifold)
Beta = (gamma_I*Cost*l / n^2); 

% Train
fast = true; % uses the pdist2 function in Matlab Stat and ML Toolbox
[y_hat, LatLapMEDmodel] = LatLapMED(X, y_obvs, GEM_args, Lap_args, Kern_args, Phi, Beta, Cost, fast);
GetRates(utility, y_hat, 'LatLapMED', true);
X_range = [[min(X(:, 1)), max(X(:, 1))]; [min(X(:, 2)), max(X(:, 2))]];
[X1mesh, X2mesh, Xval] = Boundary2D(X_range, @(xx) PredictLatLapMED(xx, LatLapMEDmodel), 100);

% 1-class SVM + Random Forest (requires LibSVM and Matlab Stat and ML Toolbox)
nu_svm_model = svmtrain(ones(n, 1), X, sprintf('-q, -s 2, -t 2, -g .5, -n %f', Phi)); 
eta_1svm = abs(y_obvs); eta_1svm(nu_svm_model.sv_indices) = 1;
RF_model = TreeBagger(100, X(l_idx, :), y_obvs(l_idx));
y_hat_RF = -1*ones(n, 1); 
y_hat_RF(eta_1svm~=0) = str2double(predict(RF_model, X(eta_1svm~=0, :)));
y_hat_RF(l_idx) = y_obvs(l_idx); % labelled points are predicted as such
y_hat_RF = sign(y_hat_RF); % convert to signs    
GetRates(utility, y_hat_RF, 'nuSVM + RF', true);


% Plot where only_obvs visible labels are shown
Figure1=figure(1); clf(Figure1); 
set(Figure1, 'Color', 'w', 'Units', 'inches', 'Position', [0,0, 14, 5.6]);   
subplot(1, 2, 1);
plot(X(:,1), X(:,2), 'LineStyle', 'none', 'Marker','o', ...
    'MarkerEdgeColor','none', 'MarkerSize', 4, 'MarkerFaceColor', [0, 0, 0]./255);
hold on;
plot(X(y_obvs==-1, 1), X(y_obvs==-1, 2), 'LineStyle', 'none', 'Marker','o', ... 
    'MarkerEdgeColor','none', 'MarkerSize', 5, 'MarkerFaceColor', [59, 196, 255]./255); 
hold on;
plot(X(y_obvs==1, 1), X(y_obvs==1, 2), 'LineStyle', 'none', 'Marker','o', ... 
    'MarkerEdgeColor','none', 'MarkerSize', 5, 'MarkerFaceColor', [245, 79, 79]./255); 
hold on
[~, boundary] = contour(X1mesh, X2mesh, Xval, 1, 'Color', [156, 191, 0]./255);
set(boundary, 'LineWidth', 1);
title('Observed Data', 'FontName', 'Times New Roman', 'FontWeight', 'Bold', 'FontSize', 20);
set(gca, 'XTick', [], 'XTickLabel',[], 'YTickLabel', [], 'YTick', []);
hold off;

% Plot as if all labels are visible
subplot(1, 2, 2);  
plot(X(:,1), X(:,2), 'LineStyle', 'none', 'Marker','o', ...
     'MarkerEdgeColor','none', 'MarkerSize', 4, 'MarkerFaceColor', [0, 0, 0]./255);
hold on;
plot(X(eta==1 & utility==0 & y_obvs==0, 1), X(eta==1 & utility==0 & y_obvs==0, 2), ... 
    'LineStyle', 'none', 'Marker','o', 'MarkerEdgeColor','none', ...
    'MarkerSize', 5, 'MarkerFaceColor', [0, 0, 150]./255); 
hold on ;
plot(X(utility==1 & y_obvs==0, 1), X(utility==1 & y_obvs==0, 2), ... 
    'LineStyle', 'none', 'Marker','o', 'MarkerEdgeColor','none', ...
    'MarkerSize', 5, 'MarkerFaceColor', [156, 9, 9]./255); 
hold on;
plot(X(y_obvs==-1, 1), X(y_obvs==-1, 2), 'LineStyle', 'none', 'Marker','o', ... 
    'MarkerEdgeColor','none', 'MarkerSize', 5, 'MarkerFaceColor', [59, 196, 255]./255); 
hold on;
plot(X(y_obvs==1, 1), X(y_obvs==1, 2), 'LineStyle', 'none', 'Marker','o', ... 
    'MarkerEdgeColor','none', 'MarkerSize', 5, 'MarkerFaceColor', [245, 79, 79]./255); 
hold on
[~, boundary] = contour(X1mesh, X2mesh, Xval, 1, 'Color', [156, 191, 0]./255); 
set(boundary, 'LineWidth', 1);
title('Complete Data', 'FontName', 'Times New Roman', 'FontWeight', 'Bold', 'FontSize', 20);
set(gca, 'XTick', [], 'XTickLabel',[], 'YTickLabel', [], 'YTick', []);
hold off;
leg = legend('Nominal', 'Invisible Low Utility', 'Invisible High Utility', ...
    'Visible Low Utility', 'Visible High Utility', 'Proposed', 'Location', 'SouthEast');
set(leg, 'FontName', 'Times New Roman', 'FontSize', 16, ...
    'Position', [0.44 0.72 0.2 0.2], 'Units', 'normalized');
leg_pts = findobj(get(leg,'children'), 'type', 'line'); set(leg_pts, 'MarkerSize', 10);
tightfig;



