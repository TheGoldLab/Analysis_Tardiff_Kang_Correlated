% % Compare fits rho to empirical rho per subject
% Produces Fig 7 sup 3
% "Normative evidence weighting and accumulation in correlated environments" 
% Tardiff et al., 2025.

 
% param_table = readtable('../Data/rho_params_best_2024-11-20.csv');
param_table = readtable('../Data/params_best_2025-01-03.csv');
empirical_table = readtable('../Data/empirical_corr_subj_2024-12-06');

subjects = unique(param_table.subject);
num_subjects = length(subjects);

% Select fit params
LrhoNeg = strcmp(param_table.param, 'Rn');
LrhoPos = strcmp(param_table.param, 'Rp');

% collect data into matrix of fit/empirical rho per condition
srdat = nans(num_subjects, 1); % save rho condition
pdat = nans(num_subjects, 2, 2); % +/- rho fits/empirical per subject

for ss = 1:length(subjects)

    % Best-fitting values of rho
    % fits = table2array(param_table(strcmp(param_table.subject, subjects{ss}), {'rho', 'value'}));
    % srdat(ss) = max(fits(:,1));
    pdat(ss,1,:) = [ ...
        table2array(param_table(strcmp(param_table.subject, subjects{ss})&LrhoNeg, 'value')), ...
        table2array(param_table(strcmp(param_table.subject, subjects{ss})&LrhoPos, 'value')), ...
        ];

    % Empirical estimates of rho
    estimates = table2array(empirical_table(strcmp(empirical_table.subject, subjects{ss}), {'rho', 'x12_corr'}));
    pdat(ss,2,:) = estimates(estimates(:,1)~=0,2);
    srdat(ss) = max(estimates(:,1));
end

% Plotz
rhos = nonanunique(srdat);
nrhos = length(rhos);
wh = 0.99.*ones(3,1);

for rr = 1:nrhos
    subplot(1,nrhos,rr); cla reset; hold on;
    plot([-1 1], [-1 1], 'k:')
    plot([-1 1], [0 0], 'k-', 'LineWidth', 0.25)
    plot([0 0], [-1 1], 'k-', 'LineWidth', 0.25)

    Lrho = srdat == rhos(rr);
    plot(pdat(Lrho,1,1), pdat(Lrho,2,1), 'kd', 'MarkerSize', 7, 'MarkerFaceColor', wh);
    plot(pdat(Lrho,1,2), pdat(Lrho,2,2), 'ks', 'MarkerSize', 7, 'MarkerFaceColor', wh);

    [Rn, Pn] = corr(pdat(Lrho,1,1), pdat(Lrho,2,1), 'type', 'Spearman');
    [Rp, Pp] = corr(pdat(Lrho,1,2), pdat(Lrho,2,2), 'type', 'Spearman');
    title(sprintf('r=%.1f\nr/p(+)=%.2f/%.2f\nr/p(-)=%.2f/%.2f', rhos(rr), ...
        Rp,Pp,Rn,Pn))
%         axis([0 7 -1 1])
end
subplot(1,nrhos,1); 
xlabel('Best-fitting rho')
ylabel('Empirical rho')


