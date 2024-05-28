function [h_psych] = psycho_plot(data,condvar,condstyle,meanvar,semvar,verbose,SNRvar)
%psych_color={'b','k','r'};

h_psych = [];
condList = unique(data.(condvar));
if ~exist('condstyle','var') || isempty(condstyle)
    condstyle = cellstr(repmat(' ',size(condList)));
end
if ischar(condstyle)
	condstyle = cellstr(repmat(condstyle,size(condList)));
end


if ~exist('verbose','var')
    verbose = false;
end

if ~exist('meanvar','var')
    meanvar = 'mean_nanmean_choice01';
end
if ~exist('semvar','var')
    semvar = 'sem_nanmean_choice01';
end
if ~exist('SNRvar','var')
    SNRvar = 'SNR';
end

%figure();
hold on;
for p=1:length(condList)
    if verbose
        sprintf('%s? %d',condvar,condList(p))
    end
    if iscell(condList)
        this_data = data(strcmp(data.(condvar),condList{p}),:);
    else
        this_data = data(data.(condvar)==condList(p),:);
    end
    
    if isempty(semvar)
        h_psych=[h_psych;plot(this_data.(SNRvar),...
            this_data.(meanvar),condstyle{p})];
    else
        h_psych=[h_psych;errorbar(this_data.(SNRvar),...
            this_data.(meanvar),...
            this_data.(semvar),condstyle{p})];
    end
end
xlim([min(data.(SNRvar))-.11, max(data.(SNRvar))+.11]);
ylim([0 1]);

end