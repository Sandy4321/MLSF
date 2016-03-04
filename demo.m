% This is an example program for the paper: 
% 
% Lu Sun et al. Multi-label classification by mining meta-label specific features. ICPR-16. 
%
% The program shows how the MLSF program (The main function is "MLSF.m") can be used.
%
% The program was developed based on the following package:
%
% LIBSVM -- A Library for Support Vector Machines
% URL: http://www.csie.ntu.edu.tw/~cjlin/libsvm

%% Make experiments repeatly
rng(1);

%% Add necessary pathes
addpath('data','eval');
addpath(genpath('func'));

%% Specify a MLC method and a dataset
method   =   'mlsf';
dataset  =   'medical';
load([dataset,'.mat']);

%% Set parameters 
svm.type      = 'Linear'; 
svm.para      = [];
% MLSF
mlsf.size     = 10;               
mlsf.epsilon  = 1e-2; 
mlsf.alpha    = 0.8;
mlsf.gamma    = 1e-2;
mlsf.rho      = 1;
% MDDM 
mddm.percent  = 30;
% LIFT
lift.ratio    = 0.1;
% LLSF
llsf.alpha             = 0.5;
llsf.beta              = 0.1; 
llsf.gamma             = 1e-2; 
llsf.maxIter           = 100;
llsf.minimumLossMargin = 1e-4;
llsf.outputtempresult  = 0;

%% Randomly select part of data
max_num = 8000;
if size(data,1) > max_num
    nRows = size(data,1); 
    nSample = max_num;
    rndIDX = randperm(nRows);
    index = rndIDX(1:nSample);
    data = data(index, :);
    target = target(:,index);
end

%% Perform n-fold cross validation
num_fold = 5; Results = zeros(5,num_fold);
indices = crossvalind('Kfold',size(data,1),num_fold);
for i = 1:num_fold
    disp(['Fold ',num2str(i)]);
    test = (indices == i); train = ~test; 
    switch method
        case 'br'
            tic; Pre_Labels = BRsvm(data(train,:),target(:,train),data(test,:),target(:,test));
        case 'mddm'
            tic; Pre_Labels = MDDM(data(train,:),target(:,train),data(test,:),target(:,test),mddm.percent);
        case 'lift'
            tic; Pre_Labels = LIFT(data(train,:),target(:,train),data(test,:),target(:,test),lift.ratio,svm);
        case 'llsf'
            tic; model_LLSF = LLSF( data(train,:),target(:,train)',llsf);
            Pre_Labels = LLSF_BR(data(train,:),target(:,train),data(test,:),target(:,test),model_LLSF,svm);
        case 'mlsf'
            tic; Pre_Labels = MLSF(data(train,:),target(:,train),data(test,:),mlsf);
    end
    Results(1,i) = toc;
    [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,target(:,test));
    Results(2:end,i) = [ExactM,HamS,MacroF1,MicroF1];
end
meanResults = squeeze(mean(Results,2));
stdResults = squeeze(std(Results,0,2) / sqrt(size(Results,2)));

%% Show the experimental results
printmat([meanResults,stdResults],[dataset,' by ',method],'Time ExactM HammingS MacroF1 MicroF1','Mean Std.');
