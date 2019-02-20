function [alpha, bias, objval] = SolveConvex(Y, eta, modded_K, C, type)

% Input arguments:
% Y - high utility = 1, low utility = -1, unlabeled = 0
% eta_hat - anomaly indicator
% modded_K - modified Kernel
% C - cost
% type - solve as a svm (libSVM) or med (cvx)
%
% Output arguments:
% alpha - support vectors/lagrange multipliers
% bias - intercept term
% objval - value of the objective at the maximum

n = size(Y, 1);
l_idx = find(Y ~= 0); l = size(l_idx, 1); % index of label pts
J = zeros(l, n); % create J matrix, l x n
for i = 1:size(l_idx, 1) 
    J(i, l_idx(i)) = 1;
end
JH = J(:, eta ~= 0); % JH = J*H, l x a
alpha = zeros(n, 1); % vector to hold alphas

JHKHJ = JH*modded_K*JH'; % only uses anomalous pts of kernel

switch type
    
    case 'svm'
        params = sprintf('-q, -s 0, -t 4, -c %f ', C); % C-SVC, precomputed kernel, cost
        pre_comp_K = [(1:size(JHKHJ, 1))' JHKHJ]; % precomputed kernel requires an index in first column
        model = svmtrain(Y(l_idx), pre_comp_K, params); % do SVM with precomputed kernel and cost C
        objval = [];
        
        % fill in non-zero anomalous alphas, model.sv_coef gives alpha.*y
        alpha(l_idx(model.sv_indices)) = model.sv_coef ./ Y(l_idx(model.sv_indices)); 
        alpha = alpha(eta ~= 0);
        bias = -model.rho; % libSVM gives neg bias for some reason

        if alpha < 0 % support vectors always postive, flip otherwise
            alpha = -alpha;
            bias = -bias;
        end
        
    case 'med'
        Y_l = Y(l_idx);
        Q = diag(Y_l)*JHKHJ*diag(Y_l); % YJK(I + LK)^-1J'Y
        try
            cvx_solver mosek % use mosek library if available
        end
        cvx_begin quiet 
            warning('off')
            cvx_expert true
            variable a(l, 1)
            maximize( sum(log(1 - a / C)) + sum(a) -.5*(a'*Q*a) ); 
            subject to 
            a >= zeros(l, 1);
            Y_l' * a == 0;     
        cvx_end
        objval = cvx_optval;
        
        alpha(l_idx) = a;
        alpha = alpha(eta ~= 0);

        % calculate intercept term
        dist_margin = Y_l - JHKHJ*diag(Y_l)*a; 
        bias = median(dist_margin(a ~= 0));
end

end

