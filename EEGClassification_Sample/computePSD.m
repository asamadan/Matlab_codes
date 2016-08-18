function  [PSD, PSD_Log, F]= computePSD(IndexWindow, X, Fs)

res = 0;
psd = cellfun(@(y) arrayfun(@(x) fft([y(IndexWindow(1,x):IndexWindow(2,x),:);zeros((res)*diff(IndexWindow(:,x)),size(y,2))])...
            /diff(IndexWindow(:,1)),1:size(IndexWindow,2),'un',0)', X, 'un',0); % normalize to length of window

%             repmat(mean(squeeze(X(y,IndexWindow(1,x):IndexWindow(2,x),:))),diff(IndexWindow(:,x))+1, 1)
%    psd = cellfun(@(x) remroverows(x, length(x)), psd,'un',0);  % fft PSD estimate
   psd = arrayfun(@(x) removerows(psd{x},length(psd{x})), 1:length(psd),'un',0);
   PSD = cellfun(@(x) mean(abs(cat(3,x{:})).^2,3), psd,'un',0);
   PSD = cellfun(@(x) x(1:floor(size(x,1)/2) + 1, :), PSD, 'un',0);
   F =  round(Fs/2)*linspace(0,1,length(PSD{1}));

   PSD_Log = cellfun(@(x) 10*log10(x), PSD, 'un', 0);
   inx = find(F>=1 & F<=30); % Retain only 2-36Hz
   PSD_Log = cellfun(@(x) x(inx,:), PSD_Log, 'un',0);
   PSD = cellfun(@(x) x(inx,:), PSD, 'un',0);
   F = F(inx);
   
   % check across all the channels 
   PSD_Log = cellfun(@(x) (x-min(x(:)))/range(x(:))+eps,PSD_Log, 'un', 0) ;
%    PSD_Log36 = cellfun(@(x) x - repmat(min(x),length(x),1) + eps, PSD_Log36, 'un',0);
end