function [outdata] = summary_stats(data, InputVars, GroupingVars,nanflag,svar,BetweenVars,ExcludeVars)
    
    %this function assumes you are already passing it aggregate data 
    %(e.g. subject averages).
    
    %within-subjects sem code adapted from: 
    %http://www.cookbook-r.com/Graphs/Plotting_means_and_error_bars_(ggplot2)/,
    %which is based on Morey (2008).
    
    if ~exist('nanflag','var') || isempty(nanflag)
        nanflag = 0;
    end
    
    if ~exist('GroupingVars','var')
        GroupingVars = {};
    end
    
    if nanflag
        meanfunc = @nanmean;
        stdfunc = @nanstd;
        semfunc = @nansem;
    else
        meanfunc = @mean;
        stdfunc = @std;
        semfunc = @sem;
    end

    means = varfun(meanfunc, data, 'InputVariables', InputVars, ...
            'GroupingVariables', GroupingVars);
	sds = varfun(stdfunc, data, 'InputVariables', InputVars, ...
    'GroupingVariables', GroupingVars);

    if exist('svar','var')
        
        if ischar(GroupingVars)
            GroupingVars = {GroupingVars};
        end
        
        if ~exist('BetweenVars','var') || isempty(BetweenVars)
            if length(GroupingVars) > 1
                warning('No BetweenVars provided. All GroupingVars treated as within subject.')
            end
            BetweenVars = {};
        end
        
        %use this to exclude variables from the normalization process.
        %Useful for say plotting a timecourse where time is not treated as
        %a factor in the analysis (i.e. running regression).
        if ~exist('ExcludeVars','var')
            ExcludeVars = {};
        end
        
        WithinVars = GroupingVars(~ismember(GroupingVars,BetweenVars) & ...
            ~ismember(GroupingVars,ExcludeVars));
        
        %might consider making this within sem block a separate function in
        %the future?
        normedData = normDataWithin(data,svar,InputVars,BetweenVars,nanflag);
        
        sems = varfun(semfunc, normedData, 'InputVariables', InputVars, ...
            'GroupingVariables', GroupingVars);
        
        %get number of levels in each within grouping factor in order to
        %compute correction; ignore missing data if they've made it this far
        nWithinGroups = varfun(@(x) length(unique(x(~ismissing(x)))),data,...
            'InputVariables',WithinVars);
        correctionFactor = sqrt(prod(nWithinGroups{1,:}) ./ ...
            (prod(nWithinGroups{1,:}) - 1));
        
        semvars = strcat([func2str(semfunc) '_'],InputVars);
        sems{:,semvars} = sems{:,semvars} .* correctionFactor;
        
    else
        sems = varfun(semfunc, data, 'InputVariables', InputVars, ...
            'GroupingVariables', GroupingVars);
    end
    
    if ~isempty(GroupingVars)
        outdata = join(means,sds,'Keys',GroupingVars,'KeepOneCopy','GroupCount');
        outdata = join(outdata,sems,'Keys',GroupingVars,'KeepOneCopy','GroupCount');
    else
        outdata = join(means,sds);
        outdata = join(outdata,sems);
    end
    
    if nnz(ismissing(outdata)) > 0
        warning(['Summary table contains missing data (e.g., NaNs). ' ...,
            'Check data carefully and considering running w/ nan flag.']);
    end

end
