function [data] = normDataWithin(data, svar, InputVars, BetweenVars,nanflag) 
% adapted from http://www.cookbook-r.com/Graphs/Plotting_means_and_error_bars_(ggplot2)/
% for removing between-subjects variance in order to compute
% within-subjects standard errors
% "Norms the data within specified groups in a data frame; it normalizes each
% subject (identified by idvar) so that they have the same mean, within each group
% specified by betweenvars."

if nanflag
    meanfunc = @nanmean;
else
    meanfunc = @mean;
end

if ischar(InputVars)
    InputVars = {InputVars};
end

if ischar(BetweenVars)
    BetweenVars = {BetweenVars};
end

GroupingVars = [svar,BetweenVars];

%get means for each subject/between subject variable
data_subjmean = varfun(meanfunc, data,'InputVariables',InputVars,...
    'GroupingVariables',GroupingVars);

meanInputVars = strcat([func2str(meanfunc) '_'], InputVars);

data = join(data, data_subjmean, 'Keys', GroupingVars,...
    'RightVariables', meanInputVars);

%normalize the variables
for v=1:length(InputVars)
    data.(InputVars{v}) = data.(InputVars{v}) - data.(meanInputVars{v}) + ...
                           meanfunc(data.(InputVars{v}));
end

%Remove the subject mean columns
data(:,meanInputVars) = [];

end


