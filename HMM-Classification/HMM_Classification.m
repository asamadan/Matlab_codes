function Rate = HMM_Classification(Data, varargin)
% This function runs a multi-class classification based on class-specific
% HMM and a maximum-likelihood classification.

% =========================================================================
% This function require Kevin Murphy's MATLAB HMM package:
%               https://www.cs.ubc.ca/~murphyk/Software/HMM/hmm.html
% to train HMMs (EM algorithm) and exectue maximum likelihood
% classification (forward algorithm)

% =========================================================================
% Here, the implementation of HMM classification is meant for continuous
% observations e.g., physical  activities (human movements), or physiological
% activities (e.g., gesture-specific forearm EMG signals)

% =========================================================================
% Inputs:
        % Data: a cell array of length N, in which each cell carries class-specific
        % samples for class n = 1, ..., N.
        
        % Class_name: a cell array with each entry carrying the name of a class
        
        % CV: cross-validation parameters:
        %     1) CV.nfolds: n-fold cross-validation (CV), default = 5
        %     2) CV.runs:   repeats CV 'runs' times, default = 10
        
% The following HMM configuration can be set:
% 1) HMM.s:        number of states,
% 2) HMM.m:        number of Guassian mixtures per state
% 3) HMM.topology: HMM topology: a) left to right ('L2R') or
%                                b) fully connected ('full')
% 4) HMM.prior:    initial state probabilities
% 5) HMM.cov:      Gaussian covariance matrices: a) diagonal ('diag') or
%                                                b) full     ('full')

% 6) HMM.AIC: if set, the function tunes for the number of states using Akaike
% information criterion (AIC) for a range of states. If HMM.AIC is set,
% then HMM.s can be a vector specifiying the range of states to be testes
% e.g., HMM.s = 2:10

% =========================================================================
% Output:
    % Rate: a structure carrying mean+/-StD k-class and binary classification
    %       rates out of cross-validation tests

%% Author: Ali Samadani
% Date:   May 31st, 2016

%% =================Set the classification paramters ======================   
    if nargin < 3
    CV.nfolds = 5; CV.runs = 10;
    end
    
    if nargin < 2
           % no class name class name is provided, then label the class 
           % sequentially, 1, 2, ... N
            Class_name = arrayfun(@num2str, 1:length(Data),'un',0);
    end
    
% Use the HMM configuration provided in the input
    try HMM = varargin{4};
    catch, HMM = [];
    end

% Set the remaining HMM parameters to default
    HMM = HMM_options(HMM);
    
% Setting the class names and the name for binary classification cases 
Names = arrayfun(@(x) arrayfun(@(y) [Class_name{x} 'vs' Class_name{y}], ...
    (x+1):length(Class_name),'un',0),1:length(Class_name),'un',0);
Names = [{[num2str(length(Class_name)) 'Class']} Names{:}];


% Devide class-specific samples into train and test samples for n-fold CV
cv = cellfun(@(y) crossvalind('kfold',length(y), CV.nfolds),Data, 'un',0);

%% ===================== Cross-validation =================================
for ii = 1:CV.runs % CVruns
    % initialize Testaccuracy that carries fold-specific classification
    % performance
    Testaccuracy = zeros(CV.nfolds, length(Names));
    
    for i = 1:CV.nfolds % n-folds CV
        % define the labels
        Labels   = arrayfun(@(x) x*ones(1,length(Data{x})),1:length(Data),'un',0);
        
        % define training and testing indices
        testind  = cellfun(@(x) x == i,cv,'un',0);
        trainind = cellfun(@not, testind,'un',0);
        
        % get the length of test datapoints, this is for binary
        % classification case
        Len = arrayfun(@(x) length(Data{x}(testind{x})), 1:length(Data));
        
        % define test labels
        TestLabel = cell2mat(cellfun(@(x,y) x(y), Labels, testind,'un',0));
        
        % =========================HMM training============================  
        e = 0.02;% this paramater is used to fatten the output Gaussian. This helps

        % if HMM.AIC is set,  different HMM configurations in terms of the number of
        % states are tested and the configuration that minimizes AIC on
        % the training data is selected
        if HMM.AIC
        m = HMM.m; % reassigning m, and topology to separate variables to avoid 
        Transition = HMM.topology;

        parfor state =s
            
            [Model{state-1}] = cellfun(@(x,y) HMMTrain(x(y), state, m,e, Transition), Data, trainind,'un',0); 
            
        end
        
        % compute  AIC
        AIC = cell2mat(cellfun(@(x) cellfun(@(y) y.AIC, x)', Model,'un',0));
        [~,AICmodel{Type}{ii,i}] = min(AIC,[],2);
        
        % return what model ended up being used based onAIC for each
        % class
        AICSelectedStates{Type}{ii,i} = arrayfun(@(x) s(x), AICmodel{Type}{ii,i});
        
        % Select a model with the min AIC
        MODEL = arrayfun(@(x) Model{AICmodel{Type}{ii,i}(x)}{x},1:size(AIC,1),'un',0);
        
        else
            MODEL = cellfun(@(x,y) HMMTrain(x(y), HMM.s, HMM.m, e, HMM.topology), Data, trainind,'un',0); % HMM for class1
        end
        
        % =========================Testing HMMs============================
        % Testing the trained class-specific HMMs for classifying test
        % samples using maximum likelihood classification
        
        Test = cellfun(@(x,y) x(y)', Data, testind,'un',0);
        Test = [Test{:}];
        
        % Max-Liklihood Classification
        % K-class classification
        [FTest]  = cell2mat(arrayfun(@(x) mhmm_logprob2(cellfun(@transpose, Test, 'Un', 0), MODEL{x}.prior, MODEL{x}.trans, ...
            MODEL{x}.mu, MODEL{x}.Sigma, MODEL{x}.obsmat), 1:length(MODEL),'Un', 0));
        [~, ClassHMMTest] = max(FTest,[],2);
        % KClass
        Testaccuracy(i,1) = sum(ClassHMMTest' == TestLabel)/length(ClassHMMTest)*100;
        
        % Binary classification- Actual labels
        TestBinary = arrayfun(@(x) arrayfun(@(y) [ones(Len(x),1); 2*ones(Len(y),1)],  ...
            (x+1):length(Len(:)), 'un',0), 1:length(Len(:)),'un',0);
        TestBinary = [TestBinary{:}];
        
        % Get the binary classification from the maximum liklihood
        % results for K class (FTest): ClassBinary
        [~,ClassBinary] = arrayfun(@(x) arrayfun(@(y) max(FTest(setxor(sum(Len(1:x-1))+(1:Len(x)), ...
                          sum(Len(1:y-1))+(1:Len(x))),[x y]), [],2), (x+1):length(Len),'un',0), 1:length(Len), 'un',0);
        ClassBinary =     [ClassBinary{:}];
        
        % compute the binary classification rates.
        Testaccuracy(i,1+(1:length(TestBinary))) = cellfun(@(x,y) sum(x == y)/length(y)*100, TestBinary,ClassBinary);
        
    end
    % Compute run-specific performance metrics:
    % Mean
    Accuracy.Mean(ii,:) = mean(Testaccuracy,1);
    
    % The run-specific variances will be used to compute pooled standard deviation
    Accuracy.Var(ii,:)  = var(Testaccuracy,[],1); 
end

% Average performance over the runs of cv
Rate.Mean = mean(Accuracy.Mean,1);
Rate.StD = sqrt(mean(Accuracy.Var,1));

% We will report mean ± pooled_standarddeviation from the above CV
arrayfun(@(x) disp([Names{x} ' Accuracy = ' sprintf('%.2f',Rate.Mean(x)) char(177) sprintf('%.2f',Rate.StD(x))]), 1:length(Rate.Mean))

end 

function HMM = HMM_options(HMM)
% This function sets the HMM parameters if not available to default

% 1) HMM.s:        number of states, default = 4
% 2) HMM.m:        number of Guassian mixtures per state, default = 3
% 3) HMM.topology: HMM topology: a) left to right ('L2R') [default] or
%                                b) fully connected ('full')
% 4) HMM.prior:    initial state probabilities, 
%                  default = state 1 is initial state
% 5) HMM.cov:      Gaussian covariance matrices: a) diagonal ('diag') [default] or
%                                                b) full     ('full')

% 6) HMM.AIC: if set, the function tunes for the number of states using Akaike
% information criterion (AIC) for a range of states. If HMM.AIC is set,
% then HMM.s can be a vector specifiying the range of states to be testes
% e.g., HMM.s = 2:10 [default if HMM.AIC is set]

if ~isfield(HMM, 's'),          HMM.s = 4;                        end
if ~isfield(HMM, 'm'),          HMM.m = 3;                        end
if ~isfield(HMM, 'topology'),   HMM.topology = 'L2R';             end
if ~isfield(HMM, 'prior'),      HMM.prior = [1 zeros(1,HMM.s-1)]; end
if ~isfield(HMM, 'cov'),        HMM.cov = 'diag';                 end
if ~isfield(HMM, 'AIC'),        HMM.AIC = 0;   
elseif length(HMM.s) == 1,      HMM.s = 2:10;                     end

end