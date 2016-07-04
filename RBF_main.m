clc
clear
close all

%% 

Algorithm='EKF';
Feature='WT';


%% Parameter Setting

c=10; % c = # of radial basis function centers.
m=3; % m = generator function parameter (integer greater than one).
gamma=0.5; % gamma = generator function parameter (typically between 0 and 1).
epsilon=0.01; % epsilon = delta-error threshold at which to stop training.

NumTrials = 20; % NumTrials = # of tests to run.  Typically about 5.
ifSaveTrainedNetwork=0; % 1-Save Trained Network to File; 0-Not Save.

switch Algorithm
    case 'SVSF'
        conv_rate=0.2; % SVSF Convergence Rate 0.1
        bound_thickness=3.1; % SVSF Boundery Thickness, 3 - FDD data, 2 - iris data
        lambda=0.5; % Pseudo Inverse Calculation
    case 'EKF'
        P0=0.1; % initial setting of estimate covariance matrix, 200 - iris data,0.1 10
        Q0=1; % initial setting of state covariance matrix, 40 - iris data,1 60
        R0=0.5; % initial setting of measurement covariance matrix, 50 - iris data,0.5 100
    case 'GD'
        eta=0.007; % gradient descent step size, 0.007 - FDD data 0.005 - iris data
end

%% Feature and Category

% Load ICE Data (You need to comment the Iris data part below)
switch Feature
    case 'WT'
        file_name_feature='Feature_average_wt_cad_cycle_1_100.mat';
    case 'FT'
        file_name_feature='Feature_average_ft_cad_cycle_1_100.mat';
    case 'CAD'
        file_name_feature='Feature_average_cad_cycle_1_100.mat';
end

%folder_name='C:\Users\Yifei Feng\Google Drive\3-Research\4-Matlab Program\FDD-NN\Temp_Data\';
folder_name='C:\Users\Yifei Feng\Desktop\FDD_Data\New_720\';
x_data_address=[folder_name,file_name_feature];
y_data_address=[folder_name,'Category_8_100.mat'];

load(x_data_address);
xall=Feature_Matrix; % xall =  features
load(y_data_address);
yall=Category_Matrix; % yall = categories.  


% % Load Iris Data (You need to comment the ICE data part above )
% xall = csvread('irisx.csv')';
% yall = csvread('irisy.csv')';


%% Define Performance Indices
AveCorrect = 0; % AveCorrect = average classification success percentage of the RBF network.
AveIter = 0; % AveIter = average # of iterations before convergence.
AveCPU = 0; % AveCPU = average CPU time before convergence.
Indices_Matrix=zeros(NumTrials,4);

%% Preparation for Constructing Training and Test Data
M=size(xall,2);
n0=size(yall,1);
MC=M/n0; % Number of Samples for one category
MT=floor(0.6*MC); % Number of Training Samples in one category 

%% Main Loop
for trial = 1 : NumTrials
    
    disp(['Trial # ',num2str(trial),' / ',num2str(NumTrials)]);

    % Construct Training Data and Test Data
    xtrain=[];
    xtest=[];
    ytrain=[];
    ytest=[];
    for i1=0:n0-1
        rand_index=randperm(MC);%
        train_index=rand_index(1:MT);
        test_index=rand_index(MT+1:end);

        x_category=xall(:,i1*MC+1:(i1+1)*MC); 
        xtrain=[xtrain, x_category(:,train_index)];
        xtest=[xtest, x_category(:,test_index)];

        y_category=yall(:,i1*MC+1:(i1+1)*MC); 
        ytrain=[ytrain, y_category(:,train_index)];
        ytest=[ytest, y_category(:,test_index)];
    end

    % Performance: 0 - Prepartion for CUP Time
    tstart = cputime;
    
    % Train an RBF network
    switch Algorithm
        case 'SVSF'
            [v, w, iters, Error_all] = RBFSVSF(xtrain, ytrain, c, gamma, m, epsilon,conv_rate, bound_thickness, lambda);
        case 'EKF'
            [v, w, iters, Error_all] = RBFKalman(xtrain, ytrain, c, gamma, m, epsilon, P0, Q0, R0);
        case 'GD'
            [v, w, iters, Error_all] = RBFGrad(xtrain, ytrain, c, gamma, m, epsilon, eta);
            %d=1;
    end
    
    % Performance: 1- Average CPU Time, 2- Average Iteration Number
    Trial_Time=cputime - tstart;
    AveCPU = ((trial-1)*AveCPU + Trial_Time) / trial;
    AveIter = ((trial-1)*AveIter + iters) / trial;
    
    
    % Training data result
    [PctCorrect_train, Err_train, yhat_train] = RBFTest(xtrain, ytrain, v, w, gamma, m);
    
    % PLot training Confusion Matrix
    figure
    plotconfusion(ytrain,yhat_train);
    

    % Test the network.
    [PctCorrect, Err, yhat] = RBFTest(xtest, ytest, v, w, gamma, m);
    
    % PLot test Confusion Matrix
    figure
    plotconfusion(ytest,yhat);
    
    % Performance: 3- Average Correction Percent
    AveCorrect = ((trial-1)*AveCorrect + PctCorrect) / trial;

    % Save Trained Network for Online Monitoring
    if(trial==NumTrials && ifSaveTrainedNetwork==1)
      folder='C:\Users\Yifei Feng\Google Drive\3-Research\4-Matlab Program\FDD-NN\Temp_Data\';
      file='RBF_network.mat';
      save([folder,file],'v','w');
    end

    % Record Performance Indices
    Indices_Matrix(trial,1)=Trial_Time;
    Indices_Matrix(trial,2)=iters;
    Indices_Matrix(trial,3)=PctCorrect;
    Indices_Matrix(trial,4)=Err; % RMS error
    
    % Plot Learning Curve
    figure
    plot(Error_all);
    title('Learning Curve');
    xlabel('Epoch');
    ylabel('Error');
    
    % Save Learning Curve Data
    file_name_learning=['LearningData_',Algorithm,'_',num2str(trial),'.mat'];
    save(file_name_learning,'Error_all');

end

%% Save Performance Indices into File
disp('last');
save_file_name=['Indices_SVSF_',num2str(c),'_hiddens_',num2str(NumTrials),'_trials.mat'];
save(save_file_name,'Indices_Matrix');