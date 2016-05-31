function RMS = computeRMS  (Signal, WindowSize, WindowShift)
% This code computes RMS values of a multi-variate 'Signal' of size Txp  
% within a moving windows of length 'WindowSize' with a shift 'WindowShift'
% Inputs: 
    % Signal: a Txp multivariate times-series observation of length T and p-
    % dimensions. 
    % WindowSize: the length ofthe moving window
    % WindowShift: the shift between subsequent windows
 
% Outputs:
% RMS: an array of RMS values obtained via moving a window of size
% 'WindowSize' along Signal with shifts of 'WindowShift'

% Author : Ali Samadani
% Data   : 26/05/2016
% =========================================================================


% defining indices of the moving windows 
IndexWindow = Move_Win_Inx(Signal, WindowSize, WindowShift);


% computing the RMS values 
RMS =  cell2mat(arrayfun(@(x) arrayfun(@(y) norm(Signal(IndexWindow(1,x):...
            IndexWindow(2,x),y))/sqrt(diff(IndexWindow(:,x))),1:size(Signal,2)),...
            1:length(IndexWindow),'uniformoutput',false)');

end