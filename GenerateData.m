
function [X, a_labels, u_labels, obvs_labels] = GenerateData(N, p, vu_percent, a_percent, type, trial_seed, samp_seed)

% Input arguments:
% N - number of samples
% p - dimension
% vu_percent - fraction of points whose utility value is visible
% a_percent - fraction of points that are considered anomalous
% type - folded t distribution or poisson distribution
% trial_seed, samp_seed - set seeds for distribution parameters and samples
%
% Output arguments:
% X - sample data points
% a_labels - anomaly = 1, nominal = 0
% u_labels - utility = 1, no utility = 0
% obvs_labels - high utility = 1, low utility = -1, unlabeled = 0

a_labels = zeros(N, 1);
u_labels = zeros(N, 1);
obvs_labels = zeros(N, 1); 

switch type

case 'folded t'

    rng(trial_seed); % set trial seed so every trial has different structure
   
    % Define distribution covariance
    Q = rand(p, p); 
    sigma = Q' * diag(abs(randn(p, 1))) * Q;
    
    % Define utility parameters
    num_u_comp = randi(4); % number of utility components
    selected = cell(num_u_comp, 1); unselected = cell(num_u_comp, 1); 
    for i = 1:num_u_comp % index of selected columns for each component
        selected{i} = datasample(1:p, randi(p-1),'Replace', false);    
        unselected{i} = setdiff(1:p, selected{i});
    end
    
    rng(samp_seed); % for fixed trial structure, get different samples
    
    % Generate samples
    X = mvtrnd(sigma, 30, N); X = abs(X);
    fX = mvtpdf(X, sigma, 15);

    % Generate utility scores for each sample
    u_score_max = NaN; 
    for i = 1:num_u_comp % difference in row means of selected and unselected columns
        u_score = mean(X(:, selected{i}), 2) - mean(X(:, unselected{i}), 2); 
        u_score_max = bsxfun(@max, u_score_max, u_score); 
    end
    
    % Create anomalous points
    a_threshold = quantile(fX, a_percent); % from CDF fX, find what the alpha cutoff is
    a_labels = fX < a_threshold; % make points less than alpha threshold anomalies
    a_idx = find(a_labels == 1);

    % Create high / low utility points from anomalies
    u_percent = 0.25; % fraction of anomalies considered to have utility
    u_score_max = u_score_max.*a_labels; % only anamolies have utility
    u_thresh = quantile(u_score_max(a_idx), 1-u_percent); % threshold for high utility 
    u_labels(a_idx) = u_score_max(a_idx) > u_thresh; 
    hu_idx = find(u_labels == 1);
    lu_idx = find(u_labels == 0 & a_labels == 1);

    % Create observed labels
    num_labeled = round((size(hu_idx, 1)*vu_percent)); 
    obvs_labels(datasample(hu_idx, num_labeled, 'Replace', false)) = 1;
    obvs_labels(datasample(lu_idx, num_labeled, 'Replace', false)) = -1;

case 'poisson'

    rng(samp_seed);
    
    X = zeros(N, 2);
    loc_func = @(vec, rep) reshape(repmat(vec, 1, rep)', size(vec, 1)*rep, 1);
    lambda_exp = p; % experiment rates

    % Define which samples are anomalous and/or high utility
    a_labels = binornd(1, a_percent, N, 1);
    u_labels = zeros(N, 1); u_labels(a_labels == 1) = binornd(1, 0.65, sum(a_labels), 1);
    
    for i = 1:N 
        
        background = [poissrnd(floor(normrnd(125, 2)), 60*6, 1); ... % night rates
                      poissrnd(floor(normrnd(135, 2)), 60*12, 1); ... % day rates
                      poissrnd(floor(normrnd(125, 2)), 60*6, 1)]; % night rates

        if a_labels(i) == 1 && u_labels(i) == 0
            c_counts1 = loc_func(binornd(1, 0.05, 60, 1), 9).*poissrnd(200, 60*9, 1); 
            c_counts2 = loc_func(binornd(1, 0.05, 60, 1), 9).*poissrnd(100, 60*9, 1); 
            course = background + [zeros(60*8, 1); c_counts1 + c_counts2; zeros(60*7, 1)];
            X(i, :) = [sum(course((60*7+1):(60*19))), sum(course)]; 
            obvs_labels(i) = binornd(1, vu_percent)*(-1);

        elseif u_labels(i) == 1
            exp_counts = loc_func(randperm(2)'-1, 60*4).*poissrnd(lambda_exp, 60*8, 1); 
            experiment = background + [zeros(60*16, 1); exp_counts];
            X(i, :) = [sum(experiment((60*7+1):(60*19))), sum(experiment)];
            obvs_labels(i) = binornd(1, vu_percent)*1;

        else
            X(i, :) = [sum(background((60*7+1):(60*19))), sum(background)]; 

        end

    end
    X = log(X);
    
end

end
