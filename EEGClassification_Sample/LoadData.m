function [Data, Class] = LoadData(Parent)
Files     = dir(Parent);
allNames  = { Files.name }; % get the list of EEG session for the param.participant
AvoidDots = cellfun(@(x)  strcmp(x, '.') | strcmp(x,'..') ,allNames);
allNames = allNames(~AvoidDots);
MatFile = allNames(cellfun(@(x) strcmp(x(end-2:end),'mat'), allNames));


% read the name of the file
C = cellfun(@(x) who('-file', [Parent '\' x]), MatFile, 'un',0);

% find classes of the data
UnderScoreIndex = cellfun(@(x) cellfun(@(y) strfind(y,'_'), x,'un',0), C, 'un',0);
% from the 2nd underscore index to end [according to the naming
% convention], indicate the class
Class = cellfun(@(x,y) cellfun(@(w,z) w(z(2)+1:end),x,y,'un',0), C, UnderScoreIndex, 'un',0);
% get rest file indices in each session to be used to combine them. 
Rest_indices = cellfun(@(x) cellfun(@(y) ~isempty(strfind(y,'Rest')), x), Class,'un',0);

% load the data 
Data = cellfun(@(x) load([Parent '\' x]), MatFile,'un',0);


% assign the data to C
% Data       = arrayfun(@(x) arrayfun(@(y) assignin('base', Class{x}{y}, getfield(Data{x}, C{x}{y})), 1:length(C{x}),'un',0),  1:length(Data),'un',0);
Data = arrayfun(@(x) arrayfun(@(y)  getfield(Data{x}, C{x}{y}), 1:length(C{x}),'un',0),  1:length(Data),'un',0);
end