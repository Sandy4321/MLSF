% This is an example program for the paper: 
% 
% Lu Sun et al. Multi-label classification with meta-label specific features. ICPR-16. 
%
% The program shows how the MLSF program (The main function is "MLSF.m") can be used.
%
% The program was developed based on the following package:
%
% LIBSVM -- A Library for Support Vector Machines
% URL: http://www.csie.ntu.edu.tw/~cjlin/libsvm

%% Make experiments repeatedly
rng(1);

%% Add necessary pathes
addpath('data','eval');
addpath(genpath('func'));

%% Choose a dataset
dataset  =   'medical';
load([dataset,'.mat']);

%% Set parameters 
opts.size     = 10;               
opts.epsilon  = 1e-2; 
opts.alpha    = 0.8;
opts.gamma    = 1e-2;
opts.rho      = 1;

%% Perform n-fold cross validation
num_fold = 5; Results = zeros(5,num_fold);
indices = crossvalind('Kfold',size(data,1),num_fold);
for i = 1:num_fold
    disp(['Fold ',num2str(i)]);
    test = (indices == i); train = ~test; 
    tic; Pre_Labels = MLSF(data(train,:),target(:,train),data(test,:),opts);
    Results(1,i) = toc;
    [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,target(:,test));
    Results(2:end,i) = [ExactM,HamS,MacroF1,MicroF1];
end
meanResults = squeeze(mean(Results,2));
stdResults = squeeze(std(Results,0,2) / sqrt(size(Results,2)));

%% Show the experimental results
printmat([meanResults,stdResults],dataset,'Time ExactM HammingS MacroF1 MicroF1','Mean Std.');
