# LatLapMED
Code for the paper: Latent Laplacian Maximum Entropy Discrimination for Detection of High-Utility Anomalies. E. Hou, K. Sricharan, A. O. Hero. IEEE Transactions on Information Forensics and Security (2018)

# Requirements
Required: CVX for optimization of the LatLapMED model 
Optional: LibSVM and Matlab's Statisitics and Machine Learning Toolbox for comparative methods in examples
Optional: Matlab's Statisitics and Machine Learning Toolbox for pdist2 function (optional input)
Optional: LibSVM if solving the quadratic dual as a relaxation using SMO

# Files
LatLapMED.m implements the Latent Laplacian Maximum Entropy Discrimination model
PredictLatLapMED.m uses a trained LatLapMED model to predict for new data
SolveConvex.m solves the quadratic dual objective of LatLapMED with a SDP or relaxed with a SMO
KernalFunc.m creates the non-laplacian gram matrix for kernels (linear, polynomial, rbf, sigmoid, cosine, tf-idf)
CreateLaplacian.m creates a graph Laplacian with distances (euclidian, cosine, angular)
GEM.m implements the Geometric Entropy Minimization algorithm with penalized distances (euclidian, cosine, angular)
GenerateData.m generates synthetic data from a folded t distribution with 30 df or poisson distributions
GetRates.m calculates the confusion matrix statistics and rates
Boundary2D.m calcuates the decision boundary of the LatLapMED model for plotting
example_2D.m runs a simulation with 2 variables and visualizes it
example_roc.m runs multiple trials of a simulation and plots the ROC curves

