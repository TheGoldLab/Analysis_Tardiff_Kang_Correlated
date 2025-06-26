% % Compare fits to mean RTs per subject
% Produces Fig 7 sup 2
% "Normative evidence weighting and accumulation in correlated environments" 
% Tardiff et al., 2025.


% 
data_table = readtable('../Data/all_parsed_data_2023-06-27.csv');
param_table = readtable('../Data/params_best_2025-01-03.csv');
% param_table = readtable('../Data/rho_params_best_2024-11-20.csv');
% ndtparam_tabe = readtable('../Data/params_best_2024-11-20.csv');

subjects = unique(data_table.subject);
num_subjects = length(subjects);

LrhoNeg = strcmp(param_table.param, 'Rn');
LrhoPos = strcmp(param_table.param, 'Rp');

% Colors
%hex2rgb = @(v) [double(hex2dec(v(1:2)))/255 double(hex2dec(v(3:4)))/255 double(hex2dec(v(5:6)))/255];
%colors = [hex2rgb('7fc97f'); hex2rgb('beaed4'); hex2rgb('fdc086')];

% collect data into matrix of conditions/mean RTs
rdat = nans(num_subjects, 1); % save rho condition
sdat = nans(num_subjects, 3, 2, 2); % rhos / mus / mean±sem RT
pdat = nans(num_subjects, 2); % +/- rho fits

for ss = 1:length(subjects)

    sdata = table2array(data_table(strcmp(data_table.subject, subjects{ss}), {'rho', 'mu', 'RT', 'correct'}));
    % ndt= table2array(ndtparam_tabe(strcmp(ndtparam_tabe.subject, subjects{ss})&strcmp(ndtparam_tabe.param, 'nondectime'), {'value'}));
    sdata = sdata(sdata(:,4)==1, 1:3);
    sdata(:,2) = abs(sdata(:,2)); % fold two directions    
    rs = unique(sdata(:,1));
    rdat(ss) = max(rs);
    for rr = 1:length(rs)
        Lr = sdata(:,1)==rs(rr);
        mus = unique(sdata(Lr,2));
        for mm = 1:length(mus)
            Lrm = Lr & sdata(:,2)==mus(mm);
            sdat(ss,rr,mm,:) = [mean(sdata(Lrm,3)) sem(sdata(Lrm,3))];
        end
    end

    pdat(ss,:) = [ ...
        table2array(param_table(strcmp(param_table.subject, subjects{ss})&LrhoNeg, 'value')), ...
        table2array(param_table(strcmp(param_table.subject, subjects{ss})&LrhoPos, 'value')), ...
        ];
    % pdata = table2array(param_table(strcmp(param_table.subject, subjects{ss}), {'rho', 'value'}));
    % pdat(ss,:) = [pdata(pdata(:,1)<0, 2), pdata(pdata(:,1)>0, 2)];    
end

% Plotz
rhos = nonanunique(rdat);
nrhos = length(rhos);
wh = 0.99.*ones(3,1);

for mm = 1:2
    for rr = 1:nrhos
        subplot(2,nrhos,(mm-1)*4+rr); cla reset; hold on;
        plot([0 8], rhos(rr).*([1 1]), 'r-');
        plot([0 8], -rhos(rr).*([1 1]), 'r-');
        plot([0 8], [0 0], 'k:')

        Lrho = rdat == rhos(rr);
        plot(sdat(Lrho,2,mm,1), pdat(Lrho,1), 'kd', 'MarkerFaceColor', wh, 'MarkerSize', 7)
        plot(sdat(Lrho,2,mm,1), pdat(Lrho,2), 'ks', 'MarkerFaceColor', wh, 'MarkerSize', 7)
        [Rn,Pn] = corr(sdat(Lrho,2,mm,1), pdat(Lrho,1), 'type', 'Pearson');
        [Rp,Pp] = corr(sdat(Lrho,2,mm,1), pdat(Lrho,2), 'type', 'Pearson'); 
        % lsline
        title(sprintf('r=%.1f\nr/p(+)=%.2f/%.2f\nr/p(-)=%.2f/%.2f', rhos(rr), ...
            Rp,Pp,Rn,Pn))
        axis([0 7 -1 1])
    end
end
subplot(2,nrhos,5); 
xlabel('RT (sec)')
ylabel('Best-fitting rho')


