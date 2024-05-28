function [sdata] = make_smooth_data(data,conds,smoothvars,smoothvar_names)
    if ischar(smoothvar_names)
        smoothvar_names = {smoothvar_names};
    end
    if ischar(conds)
        conds = {conds};
    end
    if length(conds) > 1
        warning('Multiple conditions only supported for non-numeric conditions!');
    end
    if iscell(data{:,conds}) && ~iscellstr(data{:,conds})
        %need to convert to cellstrings to get unique condition values
        condname = data(:,conds).Properties.VariableNames;
        numdata = cellfun(@(x) num2str(x),data{:,conds},'UniformOutput',false);
        numdata = unique(numdata);
        data_conds = arrayfun(@(x) str2double(strsplit(x{1})),numdata,'UniformOutput',false);
        data_conds = table(data_conds,'VariableNames',condname);
    else
        data_conds = unique(data(:,conds),'rows');
    end
    sdata = [];
    for i=1:height(data_conds)
        this_smooth_data = repmat(data_conds(i,:),size(smoothvars,1),1);
        for v=1:size(smoothvars,2)
            this_smooth_data.(smoothvar_names{v}) = smoothvars(:,v);
        end
        sdata = [sdata;this_smooth_data];
    end
    %sdata = table2mat(sdata);

end

%this is how you need to handle the pretone unique...urg
%{
foov = data(:,conds).Properties.VariableNames;
foo = data{:,conds};
foo2 = cellfun(@(x) num2str(x),foo,'UniformOutput',false);
foo3 = unique(foo2);
foo4 = arrayfun(@(x) str2double(strsplit(x{1})),foo3,'UniformOutput',false);
foo5 = table(foo4,'VariableNames',foov);
%}