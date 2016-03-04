function Pre_Labels = MDDM( train_data,train_target,test_data,test_target,percent )
%MDDM Multi-label Dimension reduction via Dependence Maximization 

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

