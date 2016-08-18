% filip classification 
% Author: Ali Samadani
% Date: Aug. 17, 2016

clear, close, clc

dbstop if error
Parent    = 'C:\Users\samadania\PRISM\Students\Filip\Filip Pilot Data';

[Data, Class] = LoadData(Parent);

%% classification options
options.kfold                        = 5;
options.runs                         = 10;
options.CSPdim                       = 3;
options.classification               = 'UL_LR';
options.Fs                           = 256;
options.CSPtype                      = 1;

% to perform featureType-specific normalization
options.FeatureSpecific_Normlization = 1;

% feature-specific unity (0-1) normalization prior to feature selection and classification
options.FeatureNormalize             = 1;

% can be set to combine the desired features
options.combine                      = 1:10; 

options.FeatureSelection             = 1;
% classifier
options.classifier                   = 'Logistic';
%% classication

[Rate, Features] = ClassificationFcn(Data, Class, options)

%% Average rates
disp(['Average Accuracy   : ' sprintf('%.2f', mean([Rate(:).Accuracy]*100)) char(177) sprintf('%.2f',  std([Rate(:).Accuracy]*100))])
disp(['Average Sensitivity: ' sprintf('%.2f', mean([Rate(:).Sensitivity]*100)) char(177) sprintf('%.2f',  std([Rate(:).Sensitivity]*100))])
disp(['Average Specificity: ' sprintf('%.2f', mean([Rate(:).Specificity]*100)) char(177) sprintf('%.2f',  std([Rate(:).Specificity]*100))])
disp(['Average AUC        : ' sprintf('%.2f', mean([Rate(:).AUC]*100)) char(177) sprintf('%.2f',  std([Rate(:).AUC]*100))])

