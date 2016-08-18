function [Train, Test, Train_Label] = Cov_CSP(Train,Test, Train_Label, N)

% Author: Ali Samadani
% Date: Aug. 17, 2016

    Train = arrayfun(@(x) Train(Train_Label == x,:,:), unique(Train_Label), 'un',0);
    COV = cellfun(@(x) arrayfun(@(y) cov(squeeze(x(y,:,:))), 1:size(x, 1), 'un',0), Train, 'un',0); 
    
    R1 = cellfun(@(x) mean(cat(3,x{:}),3),COV,'un',0);
    R = permute( cat(3,R1{:}),[3 1 2]);


    W = MulticlassCSP(R,N);
    Train       = cellfun(@(x) arrayfun(@(y) squeeze(x(y,:,:))*W',1:size(x,1),'un',0),Train, 'un',0);
    
    Train_Label = arrayfun(@(x) x*ones(1,length(Train{x}))-1, 1:length(Train), 'un',0);
    
    Test        = arrayfun(@(y) squeeze(Test(y,:,:))*W',1:size(Test,1),'un',0);
    Train       = [Train{:}];
    Train_Label = [Train_Label{:}];
    
end