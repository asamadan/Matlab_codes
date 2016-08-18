function [Rate, selectedFeatures] = ClassificationFcn(X, Class, options)

% Author: Ali Samadani
% Date: Aug. 17, 2016
rng('shuffle')

if strcmp(options.classification, 'Rest')
    Class = cellfun(@(x) cellfun(@(y) any(strfind(y, 'Rest'))+2*isempty(strfind(y, 'Rest')), x), Class, 'un',0);

elseif  strcmp(options.classification, 'UL_LR')%% expand this to include all the cases
    
    Class = cellfun(@(x) cellfun(@(y) strcmp(y, 'UL')+2*strcmp(y, 'LR'), x), Class, 'un',0);
    % have an equal participation from different sessions and different
% subsession
end

    % have an equal participation from different sessions and different
% subsession
CVO = cellfun(@(x) cellfun(@(y) cvpartition(size(y,1),'kfold',options.kfold), x, 'un',0), X, 'un',0);

% k-fold CV
% You should repeat the k-fold cross-validation for a number of times e.g.,
% 10

for fold = 1:options.kfold
    % get the indices for train and test sets, keeping the propotion of
    % data from each class and and each session equal in the train and test
    % sets
    TrainInd = cellfun(@(x) cellfun(@(y) y.training(fold),x,'un',0), CVO,  'un',0);
    TestInd  = cellfun(@(x) cellfun(@(y) ~y, x, 'un',0), TrainInd, 'un',0);
    
    Train       = cellfun(@(x,y,v) cellfun(@(w,z) w(z,:,:),x(v~=0),y(v~=0),'un',0), X, TrainInd,Class, 'un',0);
    Train_Label = cellfun(@(x,y) arrayfun(@(z) z*ones(1,size(x{z},1))-1, 1:length(find(y)),'un',0), Train, Class, 'un',0);
    
    Test        = cellfun(@(x,y,v) cellfun(@(w,z) w(z,:,:),x(v~=0),y(v~=0),'un',0), X, TestInd, Class, 'un',0);
    Test_Label  = cellfun(@(x,y) arrayfun(@(z) z*ones(1,size(x{z},1))-1, 1:length(find(y)),'un',0), Test, Class, 'un',0);
    
    if options.CSPtype
        % apply CSP
        % since the first session is shorter, we cut it out
        Train(1) = []; Train_Label(1) = [];
        Test(1)  = []; Test_Label(1)  = [];
        
        % combine train from different sessions
        Train       = cell2mat(cellfun(@(x) cell2mat(x'), Train, 'un',0)');
        Train_Label = cell2mat(cellfun(@(x) cell2mat(x), Train_Label, 'un',0));
        
        Test        = cell2mat(cellfun(@(x) cell2mat(x'), Test, 'un',0)');
        Test_Label  = cell2mat(cellfun(@(x) cell2mat(x), Test_Label, 'un',0));
        
        addpath(genpath('C:\samadania\PRISM\MulticlassCSP'))
        [Train, Test, Train_Label] = Cov_CSP(Train, Test, Train_Label, options.CSPdim);
        rmpath(genpath('C:\samadania\PRISM\MulticlassCSP'))
    else %under construction!!!
        % combine train from different sessions
        Train       = cellfun(@(y) cellfun(@(x) arrayfun(@(z) squeeze(x(z,:,:)), 1:size(x,1), 'un',0), y, 'un',0), Train, 'un',0);
        Test        = cellfun(@(y) cellfun(@(x) arrayfun(@(z) squeeze(x(z,:,:)), 1:size(x,1), 'un',0), y, 'un',0), Test, 'un',0);
        
        Train = cellfun(@(x) [x{:}], Train,'un',0); Train = [Train{:}];
        Test  = cellfun(@(x) [x{:}], Test, 'un',0); Test  = [Test{:}];
        
        Train_Label = cellfun(@(x) cell2mat(x), Train_Label, 'un',0);
        
        Test        = cell2mat(cellfun(@(x) cell2mat(x'), Test, 'un',0)');
        Test_Label  = cell2mat(cellfun(@(x) cell2mat(x), Test_Label, 'un',0));
    
    end
    
    % compute features
    
    [Train, FeatureName] = computeFeatures(Train, options);
    [Test ]              = computeFeatures(Test, options);
    
    % feature-specific normalization
    if options.FeatureSpecific_Normlization
        Min   = cellfun(@(x) min(x(:)), Train, 'un',0);
        Range = cellfun(@(x) range(x(:)), Train, 'un',0);
        
        Train = cellfun(@(x, y, z) (x - y)/z, Train, Min, Range, 'un',0);
        Test  = cellfun(@(x, y, z) (x - y)/z, Test, Min, Range, 'un',0);
        
    end
    
    % combine feature
    if ~isempty(options.combine)
        Train = cell2mat(Train(options.combine));
        Test  = cell2mat(Test(options.combine));
    end
    
% feature selection - ELASTIC NET        
        
        if options.FeatureNormalize
            [Train, Test] = Normalzation(Train,Test);
            
        end
        %
        
        if options.FeatureSelection
         [selectedFeatures{fold}, Lambda, Alpha] = ElasticNet(Train, Train_Label);
        % represent Train and Test in terms of the selected features
        Train = Train(:, selectedFeatures{fold});
        Test  = Test(:, selectedFeatures{fold});
        else 
            
            Lambda = 0; Alpha = 0;
            selectedFeatures{fold} = [];
        end        
        
        [~,  ProbTest, ClassTest] = ClassificationModule(Train, Train_Label', Test, Test_Label, options.classifier, Lambda,Alpha);
    
    


    Rate(fold).Accuracy  =  sum(Test_Label' == ClassTest)/length(ClassTest);
    TP = sum(Test_Label == 1 & ClassTest' == 1);
    TN = sum(Test_Label == 0 & ClassTest' == 0);
    FP = sum(Test_Label == 0 & ClassTest' == 1);
    FN = sum(Test_Label == 1 & ClassTest' == 0);
    
    Rate(fold).Sensitivity = TP/(TP+FN);
    Rate(fold).Specificity = TN/(TN+FP);
    Rate(fold).Precision   = TP/(TP+FP);
    [~,~,~, Rate(fold).AUC] = perfcurve(Test_Label,ProbTest(:,2),1);
    


end

end
%%
function [Train, Test] = Normalzation(Train, Test)
% feature-specific normalization
maxTrain = max(Train);
minTrain = min(Train);

Train = (Train - repmat(minTrain, ...
    size(Train,1), 1))./repmat(maxTrain-minTrain, ...
    size(Train, 1), 1);
Test = (Test - repmat(minTrain, ...
    size(Test,1), 1))./repmat(maxTrain-minTrain, ...
    size(Test, 1), 1);


end

function [selectedFeatures, Lambda, Alpha] = ElasticNet(Train, Train_Label)
% Elastic net feature selection
%    feature('jit',0)
%    feature('accel',0)
   nfolds = 10;
   pars = struct('big',1e35);
   glmnetControl(pars); 
% not used in the current implementation
d = 1; BestDev = 1;
a = 1;
for a = 0.1:0.1:1
    options.alpha = a;

    CVerr(d) = cvglmnet(Train, Train_Label,'binomial',options,'deviance',nfolds);
    basicIndx = CVerr(d).lambda == CVerr(d).lambda_1se;
    if d == 1
        BestDev = CVerr(d).cvm(basicIndx);
    end
    
      if CVerr(d).cvm(basicIndx) <= BestDev
                        if isempty(find(CVerr(d).glmnet_fit.beta(:,basicIndx)))
                            basicIndx = CVerr(d).lambda == CVerr(d).lambda_min;
                        end
                        if isempty(find(CVerr(d).glmnet_fit.beta(:,basicIndx)))
                            basicIndx = find(CVerr(d).lambda == CVerr(d).lambda_min) + 2;
                        end
                        selectedFeatures = find(CVerr(d).glmnet_fit.beta(:,basicIndx));
                        BestDev = CVerr(d).cvm(basicIndx);
                        Lambda = CVerr(d).lambda(basicIndx);
                        Alpha = a;

      end
                    
                    
    
    d = d+1;
end
end