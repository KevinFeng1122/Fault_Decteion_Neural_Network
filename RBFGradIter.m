function [AveCorrect, AveIter, AveCPU] = RBFGradIter(xall, yall, c, gamma, m, eta, epsilon)

% [AveCorrect, AveIter, AveCPU] = RBFGradIter(xall, yall, c, gamma, m, eta, epsilon)
% Test a Radial Basis Function network trained via gradient descent.
% Edit the variable NumTrials in this m-file in order to change
% the number of trials.
%
% INPUTS
% xall = 4 x 150 array of Iris features.  If this array is empty
%        then the m-file will attempt to read it from irisx.csv.
% yall = 3 x 150 array of Iris categories.  If this array is empty
%        then the m-file will attempt to read it from irisy.csv.
% c = # of radial basis function centers.
% gamma = generator function parameter (typically between 0 and 1).
% m = generator function parameter (integer greater than one).
% eta = gradient descent step size.
% epsilon = delta-error threshold at which to stop training.
%
% OUTPUTS
% AveCorrect = average classification success percentage of the
%              RBF network.
% AveIter = average # of iterations before convergence.
% AveCPU = average CPU time before convergence.

% This m-file was used to produce the results that are plotted in
% Dan Simon's paper as submitted to the journal Neurocomputing, 
% submitted in January 2000.

% NumTrials = # of tests to run.  Typically about 5.
NumTrials = 5;

if size(xall) ~= [4 150]
  disp('Reading Iris Features...');
  xall = csvread('irisx.csv')';
end
if size(yall) ~= [3 150]
  disp('Reading Iris Categories...');
  yall = csvread('irisy.csv')';
end

rand('seed', sum(100*clock));
AveCorrect = 0;
AveIter = 0;
AveCPU = 0;

%------------------------------------------------------------------------
Indices_Matrix=zeros(NumTrials,4);

%------------------------------------------------------------------------

for trial = 1 : NumTrials
  disp(['Trial # ',num2str(trial),' / ',num2str(NumTrials)]);
  % Create an array that contains 75 integers, including
  % 25 random integers between 1 and 50, 25 random integers
  % between 51 and 100, and 25 random integers between 101 and 150.
  traincol = [];
  for category = 1 : 3
    while size(traincol, 2) < 25 * category
      candidate = 1 + floor(50 * (rand(1)-eps)) + 50 * (category - 1);
      % If the random integer is not yet in the traincol array, put it in.
      Flag = 1;
      for j = 25 * (category - 1) + 1 : size(traincol, 2)
        if candidate == traincol(j)
          Flag = 0;
          break;
        end
      end
      if Flag == 1
        traincol = [traincol candidate];
      end
    end
  end
  % Sort the traincol array in ascending order.
  traincol = sort(traincol);
  % Now extract the training set from xall and yall based on
  % the integers in the traincol array.
  xtrain = [];
  ytrain = [];
  for j = 1 : 75
    xtrain = [xtrain xall(:, traincol(j))];
    ytrain = [ytrain yall(:, traincol(j))];
  end
  % Now extract the test set from xall and yall based on
  % the integers that are not in the traincol array.
  xtest = [];
  ytest = [];
  trainindex = 1;
  for j = 1 : 150
    if trainindex > size(traincol, 2)
      xtest = [xtest xall(:, j)];
      ytest = [ytest yall(:, j)];
    elseif j == traincol(trainindex)
      trainindex = trainindex + 1;
    else
      xtest = [xtest xall(:, j)];
      ytest = [ytest yall(:, j)];
    end
    if size(xtest, 2) == 75, break, end;
  end
  % Train an RBF network via gradient descent.
  tstart = cputime;
%   [v, w, iters] = RBFGrad(xtrain, ytrain, c, gamma, m, eta, epsilon);
  [v, w, iters] = RBFLM2(xtrain, ytrain, c, gamma, m, eta, epsilon);
  %------------------------------------------------------------------------
  Trial_Time=cputime - tstart;
  AveCPU = ((trial-1)*AveCPU + Trial_Time) / trial; %
  %------------------------------------------------------------------------
  % AveCPU = ((trial-1)*AveCPU + cputime - tstart) / trial;
  
  AveIter = ((trial-1)*AveIter + iters) / trial;
  % Test the network.
  [PctCorrect, Err] = RBFTest(xtest, ytest, v, w, gamma, m);
  AveCorrect = ((trial-1)*AveCorrect + PctCorrect) / trial;
  
  %------------------------------------------------------------------------
  Indices_Matrix(trial,1)=Trial_Time;
  Indices_Matrix(trial,2)=iters;
  Indices_Matrix(trial,3)=PctCorrect;
  Indices_Matrix(trial,4)=Err; % RMS error
  %------------------------------------------------------------------------
  
end
%------------------------------------------------------------------------
save_file_name=['Indices_Gradient_',num2str(c),'_hiddens_',num2str(NumTrials),'_trials.mat'];
save(save_file_name,'Indices_Matrix');
%------------------------------------------------------------------------