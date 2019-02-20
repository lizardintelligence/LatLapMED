function L = CreateLaplacian(X, L_args, use_pdist)

% Input arguments:
% X - data points
% L_args - distance metric, # of nearest neighbors, heat kernel width
% use_pdist - if true, use Matlab's pdist2 function
%
% Output arguments:
% L - Laplacian matrix

n = size(X, 1); 
L_args = strsplit(L_args); type = L_args(1); 
num_k = str2double(L_args(2)); num_k = min(n-1, num_k);
heat_sig = str2double(L_args(3)); 
k_range = 2:(num_k+1);

switch type{1}

case 'euclid' 
    if use_pdist == false
        normX = sum(X.^2, 2); 
        dist = sqrt(normX*ones(n, 1)' - 2*(X*X') + ones(n, 1)*normX');
    else
        dist = pdist2(X, X, 'euclidean'); 
    end
    
case 'cosine'
    if use_pdist == false
        normX = sum(X.^2, 2); 
        dot_prod = (X*X');
        sim = dot_prod ./ (sqrt(normX*normX'));
        sim( isnan(sim) & dot_prod==0 ) = 0; % fix so 0/0 = 0
        dist = 1 - sim;
    else
        dist = pdist2(X, X, 'cosine');   
    end

case 'angular'
    if use_pdist == false
        normX = sum(X.^2, 2); 
        dot_prod = (X*X');
        sim = dot_prod ./ (sqrt(normX*normX'));
        sim( isnan(sim) & dot_prod==0 ) = 0; % fix so 0/0 = 0   
    else
        sim = 1 - pdist2(X, X, 'cosine'); sim(isnan(sim)) = 0; % fix so 0/0 = 0
    end
    dist = 2*acos(sim)/pi; % use angular distance

end   
[kNN_vals, kNN_idx] = sort(dist, 2);
    
% create sparse heat kernel weighted adjacency graphs and fill in values
try % faster but does not allocate the correct size if all indexes are 0
    W = sparse(repmat(1:n, num_k, 1)', kNN_idx(:, k_range), exp(-kNN_vals(1:n, k_range).^2 / (4*heat_sig)));
    W = max(W, W');
catch
    W = spalloc(n, n, n*num_k);
    for i = 1:n
        W(i, kNN_idx(i, k_range)) = exp(-kNN_vals(i, k_range).^2 / (4*heat_sig));
    end
    W = max(W, W');
end

diag_D = sum(W, 2); % diag of degree matrix is col sum of weight matrix     
nW = W ./ sqrt(diag_D * diag_D'); % normalize weight matrix D^-.5 W D^-.5
nW(logical(eye(n))) = 0; % prevent issues with diag being nan or inf
L = speye(n, n) - nW; % normalized laplacian is I - nW, nW is normalized weight matrix

end

