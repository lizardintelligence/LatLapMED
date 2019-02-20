function [X1, X2, mesh_val] = SVM_Boundary2D(x_supp, pred_func, pts)

% Input arguments:
% x_supp - support of the distribution of X
% pred_func - function to predict for the mesh
% pts - number of points in the mesh
%
% Output arguments:
% X1, X2 - mesh grid points
% mesh_val - the 0/1 prediction value at each mesh point

% Build a mesh of points and get 0/1 at each point
x1plot = linspace(x_supp(1, 1), x_supp(1, 2), pts)';
x2plot = linspace(x_supp(2, 1), x_supp(2, 2), pts)';
[X1, X2] = meshgrid(x1plot, x2plot);
mesh_val = zeros(pts, pts);

for i = 1:pts
   mesh_X = [X1(i, :)', X2(i, :)']; % X1, X2 pair of coordinates
   mesh_pred = pred_func(mesh_X); mesh_pred(mesh_pred < 0) = 0;
   mesh_val(i, :) = mesh_pred;
end

end

