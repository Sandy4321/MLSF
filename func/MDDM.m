function Pre_Labels = MDDM( train_data,train_target,test_data,test_target,percent )
%MDDM Multi-label Dimension reduction via Dependence Maximization [1]
%
%    Syntax
%
%       Pre_Labels = MDDM(train_data,train_target,test_data,opts)
%
%    Description
%
%       Input:
%           train_data       An N x D data matrix, each row denotes a sample
%           train_target     An L x N label matrix, each column is a label set
%           test_data        An Nt x D test data matrix, each row is a test sample
%           percent          Dimensionality of the feature subspace
% 
%       Output:
%           Pre_Labels       An L x Nt predicted label matrix, each column is a predicted label set
%
%  [1] Y. Zhang et al. Multilabel dimensionality reduction via dependence maximization. ACM Trans. Knowl. Discov. Data, 2010.

%% Setting parameters
projtype = 'proj';
mu = 0.5;
if percent <= 1
    dim_para = round(percent * size(test_data,2));
else
    dim_para = percent;
end

%% Reduce the dimensionality of feature space by MDDM
L = train_target' * train_target; 
[P, ~] = mddm_linear(train_data',L,projtype,mu,dim_para);

%% Project the input data into feature subspaces
train_data = train_data * P;
test_data = test_data * P;

%% Use LIBSVM for classification
Pre_Labels = BRsvm(train_data,train_target,test_data,test_target);

end

