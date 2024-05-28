% fits logistic functions to choice data for 
% "Normative evidence weighting and accumulation in correlated environments" 
% Tardiff et al., 2024.

%% set options and read in data
addpath('./logistic');

model_id = 2;
DATA_DIR = './data';
OUT_DIR = 'logistic_fits';
DATA_FILE = 'all_parsed_data_2023-08-06.csv';

%read in data
data = readtable(...
    fullfile(DATA_DIR,DATA_FILE));

assert(~any(ismissing(data),'all'),'Missing data detected!!')

%% model setup
rho_labels = {'n','0','p'};
params = {'bias','SNR'};

switch model_id
    case 1 %separate fits per rho
        fit_options.base_params = params;
        fit_options.dmat_func = @make_basic_dmat;
        fit_options.fit_condvar = 'rho01'; %will fit separate per condition.
        fit_options.conds = [-1,0,1];
        fit_options.mname = 'separate';
        fit_options.model_id = model_id;
    case 2
        fit_options.base_params = params;
        fit_options.dmat_func = @make_basic_dmat;
        fit_options.mname = 'base';
        fit_options.model_id = model_id;
end
%% data setup

%add a flag for which corr you are in to help w/ separate fits
data.rho01 = sign(data.rho);

subj = unique(data(:,{'subject','rho_cond'}),'rows');

%% do fit
if isfield(fit_options,'fit_condvar')
    fits_lapse = [];
    fit_stats = [];
    for c=1:length(fit_options.conds)
        disp(fit_options.conds(c))
        this_fit_options = fit_options;

        this_data = data(data.(fit_options.fit_condvar)==fit_options.conds(c),:);        
        [this_fits_lapse,this_fit_stats] = fit_psycho_lapse(this_data,this_fit_options);
        
        this_fits_lapse = join(this_fits_lapse,subj,'Keys','subject','RightVariables','rho_cond');
        this_fit_stats = join(this_fit_stats,subj,'Keys','subject','RightVariables','rho_cond');
        
        this_fits_lapse.rho = this_fizts_lapse.rho_cond*fit_options.conds(c);
        this_fit_stats.rho = this_fit_stats.rho_cond*fit_options.conds(c);
        
        fits_lapse = [fits_lapse; this_fits_lapse];
        fit_stats = [fit_stats;this_fit_stats];
    end

else
    [fits_lapse,fit_stats] = fit_psycho_lapse(data,fit_options);
    fits_lapse = join(fits_lapse,subj,'Keys','subject');
    fit_stats = join(fit_stats,subj,'Keys','subject');
end

%check for boundary issues
fit_params = [fit_options.base_params,'lapse'];
min_param = varfun(@min,fits_lapse,'InputVariables',fit_params);
max_param = varfun(@max,fits_lapse,'InputVariables',fit_params);

if any(min_param{:,end-1} <= -299 | max_param{:,end-1} >= 299 | max_param.max_lapse>=.44)
    warning('parameter boundary hit!');
    pause(2)
end

figure();
for i=1:min(length(fit_params),16)
    subplot(4,4,i);
    histogram(fits_lapse.(fit_params{i}),25)
    title(fit_params{i});
end

near_bound=sum(abs(abs(fits_lapse{:,1:length(fit_options.base_params)})-300) <= 1);
near_bound(end+1) = sum(abs(abs(fits_lapse{:,length(fit_options.base_params)+1})-.45) <= .02);

figure();bar(near_bound);
xticklabels([fit_options.base_params,'lapse']);
xtickangle(45)


%% save output

save(fullfile(OUT_DIR,sprintf('fits_lapse_%s_%s.mat',fit_options.mname,date())),...
    'fits_lapse','fit_stats');

writetable(fits_lapse,fullfile(OUT_DIR,sprintf('fits_lapse_%s_%s.csv',...
    fit_options.mname,date())));
writetable(fit_stats,fullfile(OUT_DIR,sprintf('fit_stats_%s_%s.csv',...
    fit_options.mname,date())));