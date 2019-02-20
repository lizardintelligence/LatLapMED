function K = KernalFunc(X1, X2, k_args, use_pdist)

% Input arguments:
% X1, X2 - data points
% k_args - kernel type and parameters
% use_pdist - if true, use Matlab's pdist2 function
%
% Output arguments:
% K - kernel gram matrix

k_args = strsplit(k_args); type = k_args(1); 
k_params = str2double(k_args(2:end)); % scale, intercept, power, rbf_sig
n1 = size(X1, 1);
n2 = size(X2, 1);
  
switch type{1}

case 'linear'
    K = k_params(1) * (X1*X2') + k_params(2);

case 'poly'
    K = (k_params(1) * (X1*X2') + k_params(2)).^k_params(3);

case 'rbf' % exp(-(||Xi||^2 - 2XiXj + ||Xj||^2) / 2tau^2)
    if use_pdist == false
        normX1 = sum(X1.^2, 2); normX2 = sum(X2.^2, 2); 
        e_dist2 = normX1*ones(n2, 1)' - 2*(X1*X2') + ones(n1, 1)*normX2';
    else
        e_dist2 = pdist2(X1, X2, 'squaredeuclidean');
    end
    K = exp(-e_dist2 / (2 * k_params(4)^2)); 

case 'sigmoid'
    K = tanh(k_params(1) * (X1*X2') + k_params(2));

case 'cosine' % Xi*Xj / sqrt(||Xi||^2 ||Xj||^2)
    if use_pdist == false
        normX1 = sum(X1.^2, 2); normX2 = sum(X2.^2, 2);  
        dot_prod = (X1*X2');
        sim = dot_prod ./ (sqrt(normX1*normX2'));
        sim( isnan(sim) & dot_prod==0 ) = 0; % fix so 0/0 = 0
    else
        sim = 1 - pdist2(X1, X2, 'cosine'); sim(isnan(sim)) = 0; % fix so 0/0 = 0
    end
    K = sim;

case 'tf-idf'
    tf1 = spfun(@(xx) log(1+xx), X1); idf1 = log(n1./(1+size(X1~= 0, 1))); 
    tf2 = spfun(@(xx) log(1+xx), X2); idf2 = log(n2./(1+size(X2~= 0, 1)));
    fX1 = bsxfun(@times, tf1, idf1);
    fX2 = bsxfun(@times, tf2, idf2);
    if use_pdist == false
        normfX1 = sum(fX1.^2, 2); normfX2 = sum(fX2.^2, 2);       
        dot_prod = (fX1*fX2');
        sim = dot_prod ./ (sqrt(normfX1*normfX2')); 
        sim( isnan(sim) & dot_prod==0 ) = 0; % fix so 0/0 = 0
    else
        sim = 1 - pdist2(fX1, fX2, 'cosine'); sim(isnan(sim)) = 0; % fix so 0/0 = 0
    end
    K = sim;

case 'smooth-tf-idf'
    tf1 = spfun(@(xx) log(1+xx)+1, X1); idf1 = log(n1./(1+size(X1~= 0, 1))); 
    tf2 = spfun(@(xx) log(1+xx)+1, X2); idf2 = log(n2./(1+size(X2~= 0, 1)));
    fX1 = bsxfun(@times, tf1, idf1);
    fX2 = bsxfun(@times, tf2, idf2);
    if use_pdist == false
        normfX1 = sum(fX1.^2, 2); normfX2 = sum(fX2.^2, 2);       
        dot_prod = (fX1*fX2');
        sim = dot_prod ./ (sqrt(normfX1*normfX2')); 
        sim( isnan(sim) & dot_prod==0 ) = 0; % fix so 0/0 = 0
    else
        sim = 1 - pdist2(fX1, fX2, 'cosine'); sim(isnan(sim)) = 0; % fix so 0/0 = 0
    end
    K = sim;
    
end

end

