% Create the feature set
function [Feature, FeatureName] = computeFeatures(X,  options)

% Author: Ali Samadani
% Date: Aug. 17, 2016

    
WindowSize = options.Fs;
Trial_Len = size(X{1},1);
% Define moving window
WindowShift = round(WindowSize/10);
IndexWindow = [1:WindowShift:Trial_Len;(1:WindowShift:Trial_Len)+WindowSize];
IndexWindow(:,IndexWindow(2,:) > Trial_Len) = [];
IndexWindow(:,end+1) = [IndexWindow(1,end)+WindowShift, Trial_Len];

if options.CSPtype
CSPVar = cell2mat(cellfun(@(x) log(var(x)), X,'un',0)'); %% ** GOOD
end

[~,PSD_Log, F]= computePSD(IndexWindow,X, options.Fs);

% define bands of interest 
   Freq_interest = [{find(F>7 & F<=13)}, {find(F>13 & F<=30)}, {find(F>=7 & F<=36)}]; % put whatever range of frequencies
   Freq_interestSeq = [{find(F>=3 & F<5)}, {find(F>=5 & F<7)}, {find(F>=7 & F<9)}, ...
       {find(F>=9 & F<11)}, {find(F>=11 & F<13)}, {find(F>=13 & F<15)}, {find(F>15 & F<=17)}]; 
   
   
   BandPower = cellfun(@(x) cell2mat(arrayfun(@(y) sum(x(Freq_interest{y},:)),...
   1:length(Freq_interest), 'un', 0)'), PSD_Log, 'un',0); 
  
   BandPowerSeq = cellfun(@(x) cell2mat(arrayfun(@(y) sum(x(Freq_interestSeq{y},:)),...
   1:length(Freq_interestSeq), 'un', 0)'), PSD_Log, 'un',0);

   % 1st row of the resulting cells is alpha power (7-13Hz)
   BP = cell2mat(cellfun(@(x) x(:), BandPower, 'un', 0))'; % band power for different bands, numtrainingXchanreduc
   BPSeq = cell2mat(cellfun(@(x) x(:), BandPowerSeq, 'un', 0))'; % band power for different bands, numtrainingXchanreduc
   BPalpha = cell2mat(cellfun(@(x) x(1,:), BandPower, 'un',0)'); % alpha band power
   BPbeta  = cell2mat(cellfun(@(x) x(2,:), BandPower, 'un',0)'); % beta band power
   
   % Relative band power
   RelativeBandPower = cellfun(@(y) cell2mat(arrayfun(@(x) y(x,:)./y(end,:),1:length(Freq_interest)-1,'un',0)'), BandPower, 'un',0);
   RBP = cell2mat(cellfun(@(x) x(:), RelativeBandPower,'un',0))'; % vectorize relative band power

   % Peak frequency per band
   [Peak, MaxLoc] =  cellfun(@(y) arrayfun(@(x) max(y(Freq_interest{x},:)),1:length(Freq_interest)-1, 'un',0), PSD_Log, 'un',0);
   PeakFreqPerBand = cell2mat(cellfun(@(y) cell2mat(arrayfun(@(x) round(F(Freq_interest{x}(1) + y{x}-1)), 1:length(Freq_interest)-1,'un',0)), MaxLoc, 'un',0)');
   PeakFreqPerBand1 = cellfun(@(y) arrayfun(@(x) round(F(Freq_interest{x}(1) + y{x}-1)), 1:length(Freq_interest)-1,'un',0), MaxLoc, 'un',0);


   % Relative peak frequency power
   RPFPB = cell2mat(cellfun(@(y, z) cell2mat(arrayfun(@(x) y{x}./z(end,:), 1:length(Freq_interest)-1,'un',0)), Peak, BandPower, 'un',0)'); %relative peak freq power

   % Find index of frequency 2 hz prior to alpha peak
   FpriorAlpha = cellfun(@(x) x{1} - 2, PeakFreqPerBand1,'un',0);
   [~, FpriorAlpha_inx] = cellfun(@(x) arrayfun(@(y) min(abs(F - x(y))),1:length(x)),FpriorAlpha,'un',0);
   Q_alpha = cell2mat(cellfun(@(x, y, z) x{1}./y(z), Peak, PSD_Log,FpriorAlpha_inx,'un',0)'); % Q is ratio of peak alpha to alpha power 2 Hz prior        

   % Fractal exponent feature, fits linear line to PSD, finds the slope 
   FracExp = cell2mat(arrayfun(@(x) cell2mat(arrayfun(@(y) polyfit(log(F)', PSD_Log{x}(:,y),1),1:size(PSD_Log{x},2),'un',0)),...
        1:length(PSD_Log),'un',0)');

% Cell array of all the features, 1x9
    Feature = { CSPVar, ....
                BP, ...
                BPSeq, ...
                BPalpha, ...
                BPbeta, ...
                RBP, ...
                PeakFreqPerBand, ...
                RPFPB, ...
                Q_alpha, ...
                FracExp};
            
            
    FeatureName = {'Band power', 'Seq. band power', 'Alpha band power', 'Beta band power', 'Relative band power', 'Peak freq. per band', 'relative peak freq. power', 'Ratio of peak alpha to alpha power 2 Hz prior', 'Fractal exponent'};
    if options.CSPtype
        FeatureName = {'CSP log variance', FeatureName{:}};
    end
end

