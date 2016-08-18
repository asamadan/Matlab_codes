function [Class,  ProbTest, ClassTest] = ClassificationModule(Train, Train_Label, Test, Test_Label, Classifier, Lambda, Alpha)
% Author: Ali Samadani
% Date: Aug. 17, 2016

switch Classifier
    case 'Logistic'
        Opts.lambda = Lambda;
        Opts.alpha  = Alpha;
        B=glmnet(Train,Train_Label,'binomial', Opts);
        Prob = glmnetPredict(B,Test,[], 'response');
        ClassTest = (Prob>0.5) ;
        ProbTest(:,1) = 1 - Prob; ProbTest(:,2) = Prob;
        
        ProbTrain1 = glmnetPredict(B,Train,[], 'response');
                ProbTrain(:,1) = 1 - ProbTrain1; ProbTrain(:,2) = ProbTrain1;

        Class = (ProbTrain1>0.5) ;
        
    case 'SVM_RBF'
        Classification.classifierSVM = 'RBF';
        Classification.SVMprob= 1 ;
        
        [parsShort] = SVMTune(Train, Train_Label,Classification);
        
        Model= svmtrain(Train_Label, Train, parsShort);
        if Classification.SVMprob
            [ClassTest, ~, ProbTest] = svmpredict(Test_Label, Test, Model, '-b 1');
            [Class] = svmpredict(Train_Label, Train, Model, '-b 1');
        else
            [ClassTest] = svmpredict(Test_Label, Test, Model);
            [Class] = svmpredict(Train_Label, Train, Model);
            ProbTest = ones (size(ClassTest));
            
        end
        
    case 'SVM_Linear'
        Classification.classifierSVM = 'linear';
        Classification.SVMprob= 1 ;
        
        [parsShort] = SVMTune(Train, Train_Label,Classification);
        
        Model= svmtrain(Train_Label, Train, parsShort);
        if Classification.SVMprob
            [ClassTest, ~, ProbTest] = svmpredict(Test_Label, Test, Model, '-b 1');
            [Class] = svmpredict(Train_Label, Train, Model, '-b 1');
        else
            [ClassTest] = svmpredict(Test_Label, Test, Model);
            [Class] = svmpredict(Train_Label, Train, Model);
            ProbTest = ones (size(ClassTest));
            
        end
        
    case 'LDA'
        try
            [ClassTest,~, ProbTest] = classify(Test,Train,Train_Label,options.classifierLDA);
            [Class] = classify(Train,Train,Train_Label,options.classifierLDA);
        catch
            [ClassTest,~, ProbTest] = classify(Test,Train,Train_Label,'diaglinear');
            [Class] = classify(Train,Train,Train_Label,'diaglinear');
        end
        
    case 'QDA'
        try
            [ClassTest,~, ProbTest] = classify(Test,Train,Train_Label,options.classifierLDA);
            [Class] = classify(Train,Train,Train_Label,options.classifierLDA);
        catch
            [ClassTest,~, ProbTest] = classify(Test,Train,Train_Label,'diaglinear');
            [Class] = classify(Train,Train,Train_Label,'diaglinear');
        end
        
    case 'NB'
        
        [ClassTest,~, ProbTest] = classify(Test,Train,Train_Label,'diaglinear');
        [Class] = classify(Train,Train,Train_Label,'diaglinear');
        
    case 'Ensemble'
        % %         learner = 'linear';
        
        %  t = ClassificationDiscriminant.template('DiscrimType',learner);
        % Ensemble = fitensemble(Train,Train_Label,'LogitBoost',100,'tree');
        % Ensemble = TreeBagger(100,Train,Train_Label,'method', 'classification');
        %         [ClassTest, ProbTest] = predict(Ensemble,Test);
        %          ClassTest=  cellfun(@str2num, ClassTest);
end
end