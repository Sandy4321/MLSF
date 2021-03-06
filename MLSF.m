function Pre_Labels = MLSF( train_data,train_target,test_data,opts )
% MLSF MLC with Meta-Label Specific Features [1]
%
%    Syntax
%
%       Pre_Labels = MLSF(train_data,train_target,test_data,opts)
%
%    Description
%
%       Input:
%           train_data       An N x D data matrix, each row denotes a sample
%           train_target     An L x N label matrix, each column is a label set
%           test_data        An Nt x D test data matrix, each row is a test sample
%           opts             Parameters for CLMLC
%             opts.size      Size of meta-label
%             opts.epsilon   Epsilon-neighhood threshold of MLSF_META
%             opts.alpha     Importance factor of MLSF_META
%             opts.gamma     Sparsity parameter of MLSF_LASSO
%             opts.rho       A parameter of MLSF_LASSO
% 
%       Output:
%           Pre_Labels       An L x Nt predicted label matrix, each column is a predicted label set
%
%  [1] Lu Sun et al. Multi-label classification with meta-label specific features. ICPR-16. 

%% Set parameters
cluster_size = opts.size;
epsilon      = opts.epsilon;
alpha        = opts.alpha;
gamma        = opts.gamma;
rho          = opts.rho;

%% Get the size of data
num_label = size(train_target,1);
num_test  = size(test_data,1);

%% Meta-label learning
K = ceil(num_label/min(cluster_size,num_label));
if K > 1
    m = MLSF_META(train_data,train_target,alpha,epsilon,K);
else
    m = ones(num_label,1);
end

%% Specific features mining
V = MLSF_LASSO(train_data,train_target,K,m,gamma,rho);

%% Build classifier chains for each meta-label
Pre_Labels = zeros(num_test,num_label); 
null_target = zeros(num_test,1);
for j = 1:K 
    idx_feature = (V(:,j)~=0); idx_meta = (m==j);
    meta_train_data = train_data(:,idx_feature);
    meta_test_data = test_data(:,idx_feature);
    meta_train_target = train_target(idx_meta,:)';
    meta_size = size(meta_train_target,2);
    meta_Pre_Labels = zeros(num_test,meta_size);
    if meta_size == 1
        model = svmtrain(meta_train_target(:,1),meta_train_data,'-t 0 -q');
        meta_Pre_Labels(:,1) = svmpredict(null_target,meta_test_data,model,'-q');
    else
        chain = randperm(meta_size); 
        for k = chain             
            model = svmtrain(meta_train_target(:,k),meta_train_data,'-t 0 -q');
            meta_Pre_Labels(:,k) = svmpredict(null_target,meta_test_data,model,'-q');           
            meta_train_data = [meta_train_data meta_train_target(:,k)];
            meta_test_data = [meta_test_data meta_Pre_Labels(:,k)];      
        end
    end
    Pre_Labels(:,idx_meta) = meta_Pre_Labels;
end
Pre_Labels = Pre_Labels';

end
