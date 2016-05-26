function IndexWindow = Move_Win_Inx(Data, WindowSize, WindowShift)
% this function returns start and stop indices for a moving window of
% length 'WindowSize' moving sequentially along the length of the signal
% 'Data' with a shift of 'WindowShift' between subsequent steps.
% Inputs:
    % Data: signal of size (T, p), where T is the interval over which the
    %       signal exists, and p is the dimension of the signal. In the case 
    %       of mutil-variate time-series observations, p>1

    % WindowSize : The size of the moving window
    % WindowShift: The shift between subsequent windows

% Outputs:
    % IndexWindow: is a 2 x n array, with the 1st row storing start indices of
    % the moving window and the 2nd row carries the corresponding stop indices
    % of the moving window
        % e.g., IndexWindow(:,5) = [10;15]
        % The 5th window starts at time-stamp 10 and end at the 15th time-stamp

% Author: Ali Samadani
% Data: 2016/05/26
% =========================================================================

lengthofData = size(Data,1);
IndexWindow = [1:WindowShift:lengthofData;(1:WindowShift:lengthofData)+WindowSize];

% eliminate windows that finish beyond the length of the signal
IndexWindow(:,IndexWindow(2,:) > lengthofData) = []; 

% assign what left from the end of the signal to the last window
IndexWindow(:,end+1) = [IndexWindow(1,end)+WindowShift, lengthofData];