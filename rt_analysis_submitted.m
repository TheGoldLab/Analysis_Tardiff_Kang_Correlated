% RT analysis producing Figure 4
% "Normative evidence weighting and accumulation in correlated environments" 
% Tardiff et al., 2024.

data_table = readtable('../Data/all_parsed_data_2023-08-06.csv');
subjects = unique(data_table.subject);
num_subjects = length(subjects);

% Colors
hex2rgb = @(v) [double(hex2dec(v(1:2)))/255 double(hex2dec(v(3:4)))/255 double(hex2dec(v(5:6)))/255];
colors = [hex2rgb('7fc97f'); hex2rgb('beaed4'); hex2rgb('fdc086')];

% collect data into matrix of conditions/mean RTs
sdat = nan(num_subjects, 3, 2, 4); % rhos / mus / mu, rho, mean±sem RT
ex = 3; % example session
exdat = cell(3,2); % RTs for each rho, mu
for ss = 1:length(subjects)

    sdata = table2array(data_table(strcmp(data_table.subject, subjects{ss}), {'rho', 'mu', 'RT', 'correct'}));
    sdata = sdata(sdata(:,4)==1, 1:3);
    sdata(:,2) = abs(sdata(:,2)); % fold two directions    
    rs = unique(sdata(:,1));
    for rr = 1:length(rs)
        Lr = sdata(:,1)==rs(rr);
        mus = unique(sdata(Lr,2));
        for mm = 1:length(mus)
            Lrm = Lr & sdata(:,2)==mus(mm);
            sdat(ss,rr,mm,:) = [mus(mm) rs(rr) mean(sdata(Lrm,3)) sem(sdata(Lrm,3))];
            if ss==ex
                exdat{rr,mm} = sdata(Lrm,3);
            end
        end
    end
end

% PLOTZ

%% First row is predictions for Bound ~mu, ~mu^2, ~LLR
rhos = [-0.8 0 0.8];
llrs = 0.14;
[pmfs, cmfs, gMeans] = simulateDDM(0.1, 1, 1.0, 0, 0, ...
    'llrs', llrs, 'rs', rhos, 'gSigma', 0.1);
titles = {'Fixed bound on µ_g', 'Fixed bound on µ_g^2', 'Fixed bound on LLR'};
vals = cat(1, gMeans, gMeans.^2, gMeans.^2./(1+rhos));
for ii = 1:3
    subplot(8,3,3+ii); cla reset; hold on;
    title(titles{ii})
    plot(rhos, 1./vals(ii,:)./mean(1./vals(ii,:)), 'o-', 'Color', colors(ii,:), ...
        'MarkerFaceColor', colors(ii,:), 'LineWidth', 3);
    axis([rhos(1) rhos(end) 0 2.5])
    set(gca, 'FontSize', 14)
    if ii == 1
        xlabel('Generative correlation')
        ylabel('Relative RT')
    end
end

%% Second row is example session RTs per rho, mu
titles = {'Low SNR', 'High SNR'};
for mm = 1:2
    ax = subplot(4,2,2+mm); cla reset; hold on;
    XJitterWidth = 0.8 * min(diff(unique(sdat(ex,:,mm,2))));
    mdat = nan(3,2); % mean/sem per r
    for rr = 1:3
        swarmchart(ax, sdat(ex,rr,mm,2).*ones(size(exdat{rr,mm})), ...
            exdat{rr,mm}, 10, 0.8.*ones(1,3), 'XJitterWidth', XJitterWidth);
        mdat(rr,:) = [mean(exdat{rr,mm}), sem(exdat{rr,mm})];
    end
    plot(repmat(sdat(ex,:,mm,2),2,1), [mdat(:,1)-mdat(:,2) mdat(:,1)+mdat(:,2)]', 'k-', 'LineWidth', 3);
    plot(sdat(ex,:,mm,2), mdat(:,1), 'ko-', 'MarkerFaceColor', 'k', 'LineWidth', 3);
    xlabel('Generative correlation')
    ylabel('Response time (ms)')
    title(titles{mm})
    set(gca, 'XTick', sdat(ex,:,mm,2), 'FontSize', 14)    
end

%% Third row is mean rts per subject
for xx = 1:8
    subplot(4,8,16+xx); cla reset; hold on;
    set(gca, 'FontSize', 14)
    axis([[-0.25 0.25].*(mod(xx-1,4)+1) 0 7]);
    if xx == 1
        xlabel('Correlation')
        ylabel('Mean RT (sec)')
    end
end
for ss = 1:num_subjects
    for mm = 1:2
        subplot(4,8,16+(mm-1)*4+max(sdat(ss,:,mm,2).*10)/2);
        plot(sdat(ss,:,mm,2), sdat(ss,:,mm,3), 'k-');
    end
end

% Get slopes of linear fits to Evidence * RT vs condition index
% Normalize Evidence to mean value to make slopes on comparable scale
%   and flexible with respect to unknown scale factor
bdat = nan(num_subjects, 3, 2); % Save slope
brdat = nan(num_subjects, 1); % Save max rho per subject, mean, slope
Xfit = [ones(3,1) (1:3)'];
for ss = 1:num_subjects
    brdat(ss) = max(reshape(sdat(ss,:,:,2),[],1));
    for mm = 1:2
        mus  = sdat(ss,:,mm,1)';
        rhos = sdat(ss,:,mm,2)';
        RTs   = sdat(ss,:,mm,3)';
        Ss   = cat(2, mus, mus.^2, mus.^2./(1+rhos));
        Ss   = Ss ./ repmat(mean(Ss),3,1);
        for ii = 1:3
            Bs = Xfit\(RTs.*Ss(:,ii));
            bdat(ss,ii,mm) = Bs(2);
        end
    end
end

%% Bottom row are box & whisker plots of slopes per condition
% Model an agent that uses the normative evidence, 
% slightly underestimating rho
rhos = 0.2:0.2:0.8;
llrs = [0.14 0.89]';
rdat = nan(length(rhos), 2, 3); % for each rho, mu, evidence type
for rr = 1:length(rhos)
    rs = [-rhos(rr) 0 rhos(rr)];
    [pmfs, cmfs, gMeans] = simulateDDM(0.1, 1, 0.7, 0, 0, ...
        'llrs', llrs, 'rs', rs, 'gSigma', 0.1);
    % Compute slopes for each evidence type
    for mm = 1:2
        mus = gMeans(mm,:)';
        rts = cmfs(mm,:,1)';
        Ss = cat(2, mus, mus.^2, mus.^2./(1+rs'));
        Ss = 1./Ss ./ repmat(1./mean(Ss),3,1);
        for ii = 1:3
            Bs = Xfit\(rts.*Ss(:,ii));
            rdat(rr,mm,ii) = Bs(2);
        end
    end
end

for mm = 1:2
    subplot(4,2,6+mm); cla reset; hold on;
    plot([0 5], [0 0], 'k:');
    for ii = 1:3
        plot(1:4, rdat(:,mm,ii), ':', 'LineWidth', 3, 'Color', colors(ii,:));
    end
end

% Plot data
for mm = 1:2
    subplot(4,2,6+mm);
    plot([0 13], [0 0], 'k--');
    bb= boxplot(bdat(:,3,mm), brdat, 'Symbol', 'k.', 'Notch', 'on', 'colors', 'k');
    set(bb, 'LineWidth', 2)
    xlabel('Condition')
    ylabel('Slope')
    set(gca, 'XLim', [0 5], 'FontSize', 14)
end

