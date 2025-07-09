% RT analysis producing Figure 4
% "Normative evidence weighting and accumulation in correlated environments" 
% Tardiff et al., 2025.

data_table = readtable('../Data/all_parsed_data_2023-06-27.csv');
subjects = unique(data_table.subject);
num_subjects = length(subjects);

% Colors
hex2rgb = @(v) [double(hex2dec(v(1:2)))/255 double(hex2dec(v(3:4)))/255 double(hex2dec(v(5:6)))/255];
colors = [hex2rgb('beaed4'); hex2rgb('7fc97f'); hex2rgb('fdc086'); hex2rgb('007305')];

% collect data into matrix of conditions/mean RTs
sdat = nans(num_subjects, 3, 2, 4); % rhos / mus / mu, rho, mean±sem RT
ex = 6; % example session
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
% rhos = [-0.8 0 0.8];
% llrs = 0.14;
% [pmfs, cmfs, gMeans] = simulateDDM(0.1, 1, 1.0, 0, 0, ...
%     'llrs', llrs, 'rs', rhos, 'gSigma', 0.1);
% titles = {'Fixed bound on µ_g', 'Fixed bound on µ_g^2', 'Fixed bound on LLR'};
% vals = cat(1, gMeans, gMeans.^2, gMeans.^2./(1+rhos));
% for ii = 1:3
%     subplot(8,3,3+ii); cla reset; hold on;
%     title(titles{ii})
%     plot(rhos, 1./vals(ii,:)./mean(1./vals(ii,:)), 'o-', 'Color', colors(ii,:), ...
%         'MarkerFaceColor', colors(ii,:), 'LineWidth', 3);
%     axis([rhos(1) rhos(end) 0 2.5])
%     set(gca, 'FontSize', 14)
%     if ii == 1
%         xlabel('Generative correlation')
%         ylabel('Relative RT')
%     end
% end

%% First row is example session RTs per rho, mu
titles = {'Low SNR', 'High SNR'};
for mm = 1:2
    ax = subplot(3,2,mm); cla reset; hold on;
    plot(sdat(ex,:,mm,2), [mean(exdat{1,mm}); mean(exdat{2,mm}); mean(exdat{3,mm})], 'k.');
    h=lsline;
    XJitterWidth = 0.8 * min(diff(unique(sdat(ex,:,mm,2))));
    mdat = nans(3,2); % mean/sem per r
    for rr = 1:3
        swarmchart(ax, sdat(ex,rr,mm,2).*ones(size(exdat{rr,mm})), ...
            exdat{rr,mm}, 10, 0.8.*ones(1,3), 'XJitterWidth', XJitterWidth);
        mdat(rr,:) = [mean(exdat{rr,mm}), sem(exdat{rr,mm})];
    end
    plot(repmat(sdat(ex,:,mm,2),2,1), [mdat(:,1)-mdat(:,2) mdat(:,1)+mdat(:,2)]', 'k-', 'LineWidth', 3);
    plot(sdat(ex,:,mm,2), mdat(:,1), 'ko', 'MarkerFaceColor', 'k', 'LineWidth', 3);

    xlabel('Generative correlation')
    ylabel('Response time (ms)')
    title(titles{mm})
    set(gca, 'XTick', sdat(ex,:,mm,2), 'FontSize', 14)
    axis([-0.6 0.6 0 12])
end

%% Second row is mean rts per subject, plotted separately for each rho
%   condition, as a function of (signed) rho
for xx = 1:8
    subplot(3,8,8+xx); cla reset; hold on;
    set(gca, 'FontSize', 14)
    axis([[-0.25 0.25].*(mod(xx-1,4)+1) 0 7]);
    if xx == 1
        xlabel('Correlation')
        ylabel('Mean RT (sec)')
    end
end
for ss = 1:num_subjects
    for mm = 1:2
        subplot(3,8,8+(mm-1)*4+max(sdat(ss,:,mm,2).*10)/2);
        plot(sdat(ss,:,mm,2), sdat(ss,:,mm,3), 'k-');
    end
end

% Get differences in RT for + vs - rho
rScales = [0 0.9 1 -1];
numScales = length(rScales);
gVar = 0.01;
bdat = nans(num_subjects, 2, numScales+1); % Save values per mu, last is 1 real data, 3 sims (per rScale)
brdat = nans(num_subjects, 1); % Save max rho per subject
% Xfit = [ones(3,1) (1:3)'];
for ss = 1:num_subjects
    disp(ss)

    % Get rho for this subject and use in regression
    brdat(ss) = max(reshape(sdat(ss,:,:,2),[],1));
    % Xfit(:,2) = [-1 0 1]'.*brdat(ss);
    for mm = 1:2

        % parse data
        mus  = sdat(ss,:,mm,1)';
        rhos = sdat(ss,:,mm,2)';
        RTs   = sdat(ss,:,mm,3)';

        % Fit emprical data (rt vs rho)
        % Bs = Xfit\RTs;
        % bdat(ss,mm,1) = Bs(2);
        bdat(ss,mm,1) = RTs(3) - RTs(1);

        % Simulate using same task parameters, different rScales
        for cc = 1:numScales
            % compute bound based on RT at rho=0, 5 is because 5 Hz
            llr = 4.*mus(2).^2./gVar;
            bound = RTs(2).*llr*5;
            [pmfs, cmfs, gMeans] = simulateDDM(bound, 1, rScales(cc), 0, 0, ...
                'llrs', llr, 'rs', rhos', 'gSigma', 0.1);
            % Bs = Xfit\(cmfs(:,:,1)'./5);
            % bdat(ss,mm,1+cc) = Bs(2);
            bdat(ss,mm,1+cc) = (cmfs(1,3,1)-cmfs(1,1,1))./5;
        end
    end
end

% Plot the simulations (normative with rho, normative with mis-estimated rho,
%   normative without considering rho)
u_rhos = nonanunique(brdat);
num_rhos = length(u_rhos);
for mm = 1:2
    subplot(3,2,4+mm); cla reset; hold on;
    simdat = nan(num_rhos, numScales);
    for rr = 1:num_rhos
        Lrho = brdat == u_rhos(rr);
        simdat(rr,:) = median(bdat(Lrho,mm,1+(1:numScales)));
    end
    plot([0 5], [0 0], 'k:');
    for cc = 1:numScales
        plot(1:4, simdat(:,cc), ':', 'LineWidth', 3, 'Color', colors(cc,:));
    end
end

% Plot data as boxplots per rho condition
for mm = 1:2
    subplot(3,2,4+mm);
    plot([0 13], [0 0], 'k--');
    bb= boxplot(bdat(:,mm,1), brdat, 'Symbol', 'k.', 'Notch', 'on', 'colors', 'k');
    set(bb, 'LineWidth', 2)
    xlabel('Condition')
    ylabel('Slope')
    set(gca, 'XLim', [0 5], 'FontSize', 14)
end


