function [eta_hat, sorted_dist] = GEM(X, Y, d_hat, phi, G_args, sorted_dist, use_pdist)

% Input arguments:
% X - data points
% Y - high utility = 1, low utility = -1, unlabeled = 0
% d_hat - signed predicted distance from margin
% phi - percentage of anomalous points
% G_args - distance metric, # of nearest neighbors, threshold from margin
% sorted_dist.val, sorted_dist.idx - sorted distance matrix and its index
% use_pdist - if true, use Matlab's pdist2 function
%
% Output arguments:
% eta_hat - estimates of whether a point is anomalous
% sorted_dist.val, sorted_dist.idx - sorted distance matrix and its index

n = size(Y, 1);
G_args = strsplit(G_args); type = G_args(1); 
num_k = str2double(G_args(2)); num_k = min(n-1, num_k);
thresh = str2double(G_args(3)); % threshold of how far from margin is significant
k_range = 2:(num_k+1);

eta_hat = abs(Y); % all labeled point are anomalous
K_pts =  n*phi - sum(abs(Y)); % number of anomalies not counting labeled ones
ul_idx = find(Y == 0); ul = size(ul_idx, 1); % index of unlabeled points

penalty_idx = find(d_hat > thresh | Y == 1); % all points > threshold on + side or labels +
penalty = zeros(n, 1); penalty(penalty_idx) = d_hat(penalty_idx); 
    
% Calculate sorted distances if not given
if isempty(sorted_dist) == true
    
    switch type{1}

    case 'euclid'
        if use_pdist == false
            normX = sum(X.^2, 2); 
            dist = sqrt(normX*ones(n, 1)' - 2*(X*X') + ones(n, 1)*normX');
            [val, idx] = sort(dist, 1);
        else
            [val, idx] = pdist2(X, X, 'euclidean', 'Smallest', num_k+1); 
        end

    case 'cosine'
        if use_pdist == false
            normX = sum(X.^2, 2); 
            dot_prod = (X*X');
            sim = dot_prod ./ (sqrt(normX*normX'));
            sim( isnan(sim) & dot_prod==0 ) = 0; % fix so 0/0 = 0
            dist = 1 - sim;
            [val, idx] = sort(dist, 1);
        else
            [val, idx] = pdist2(X, X, 'cosine', 'Smallest', num_k+1); 
        end

    case 'angular'
        if use_pdist == false
            normX = sum(X.^2, 2); 
            dot_prod = (X*X');
            sim = dot_prod ./ (sqrt(normX*normX'));
            sim( isnan(sim) & dot_prod==0 ) = 0; % fix so 0/0 = 0 
            dist = 2*acos(sim)/pi;
            [val, idx] = sort(dist, 1);
        else
            [val, idx] = pdist2(X, X, 'cosine', 'Smallest', num_k+1); 
            sim = 1 - val; % Largest Similarity = Smallest Angle, arccos is 1 to 1
            val = 2*acos(sim)/pi; % use angular distance
        end

    end 
    
    sorted_dist.val = val'; sorted_dist.idx = idx';
    
end

% Get penalized kNN and create sparse power weighted edge graphs
kNN_idx = zeros(ul, num_k);
kNN_vals = zeros(ul, num_k);
for i = 1:ul
    kNN_idx(i, :) = sorted_dist.idx(ul_idx(i), k_range);
    kNN_vals(i, :) = sorted_dist.val(ul_idx(i), k_range) + penalty(kNN_idx(i, :))';
end

PwG = sparse(repmat(1:ul, num_k, 1)', kNN_idx, kNN_vals);
Sum_PwG = sum(PwG, 2); % total power weighted edge length for each node
[~, edge_idx] = sort(Sum_PwG); % sort total edge lengths
eta_hat(ul_idx(edge_idx(((ul-K_pts)+1):ul))) = 1; 

end
