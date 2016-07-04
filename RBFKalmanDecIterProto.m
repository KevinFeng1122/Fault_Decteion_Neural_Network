function [AveCorrect, AveIter, AveCPU] = RBFKalmanDecIterProto(xall, yall, cmin, cmax, gamma, m, epsilon, P0, Q0, R0)

% [AveCorrect, AveIter, AveCPU] = RBFKalmanDecIterProto(xall, yall, cmin, cmax, gamma, m, eta, epsilon)
% Test a Radial Basis Function network trained with a Decoupled Kalman filter.
% This function iterates on the number of prototypes in the RBF network.
% for each prototype count, the RBF is tested by calling the RBFKalmanDecIter function.
%
% INPUTS
% xall = 4 x 150 array of Iris features.  If this array is empty
%        then the m-file will attempt to read it from irisx.csv.
% yall = 3 x 150 array of Iris categories.  If this array is empty
%        then the m-file will attempt to read it from irisy.csv.
% cmin = minimum # of radial basis function centers.
% cmax = maximum # of radial basis function centers.
% gamma = generator function parameter (typically between 0 and 1).
% m = generator function parameter (integer greater than one).
% epsilon = delta-error threshold at which to stop training.
% P0 = initial setting of estimate covariance matrix (40?)
% Q0 = initial setting of state covariance matrix (40?)
% R0 = initial setting of measurement covariance matrix (40?)
%
% OUTPUTS
% AveCorrect = average classification success percentage of the RBF network;
%              an array with one element for each value of c.
% AveIter = average # of iterations before convergence.
%           an array with one element for each value of c.
% AveCPU = average CPU time before convergence.
%          an array with one element for each value of c.

% This m-file was used to produce the results that are plotted in
% Dan Simon's paper as submitted to the journal Neurcomputing, 
% submitted in January 2000.

if size(xall) ~= [4 150]
  disp('Reading Iris Features...');
  xall = csvread('irisx.csv')';
end
if size(yall) ~= [3 150]
  disp('Reading Iris Categories...');
  yall = csvread('irisy.csv')';
end

AveCorrect = zeros(1, cmax - cmin + 1);
AveIter = zeros(1, cmax - cmin + 1);
AveCPU = zeros(1, cmax - cmin + 1);

for c = cmin : cmax
   disp(' ');
   disp([num2str(c),' Prototypes']);
   disp(' ');
   [Correct, Iter, CPU] = RBFKalmanDecIter(xall, yall, c, gamma, m, epsilon, P0, Q0, R0);
   AveCPU(c-cmin+1) = CPU;
   AveIter(c-cmin+1) = Iter;
   AveCorrect(c-cmin+1) = Correct;
end
