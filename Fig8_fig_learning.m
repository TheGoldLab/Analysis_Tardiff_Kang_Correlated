% Compare best-fitting rho (full session) to change in best-fitting
%   rho from early to late in session
% Produces figure 8
% "Normative evidence weighting and accumulation in correlated environments" 
% Tardiff et al., 2025.

% Compare best-fitting rho (full session) to change in best-fitting
%   rho from early to late in session
halfSessionTable = readtable('../Data/params_half_best_2025-01-04.csv');
dataTable = readtable('../Data/all_parsed_data_2023-06-27.csv');
subjects = unique(dataTable.subject);
numSubjects = length(subjects);

% Selection arrays for fit data
Ldrift = strcmp(halfSessionTable.param, 'driftSNR0');
Lbound = strcmp(halfSessionTable.param, 'B0');
LrhoNeg = strcmp(halfSessionTable.param, 'Rn');
LrhoPos = strcmp(halfSessionTable.param, 'Rp');
LfirstHalf  = halfSessionTable.half == 1;

% Columns are:
%   1 ... rho
%   2 ... fit_first_half_k
%   3 ... fit_first_half_bound
%   4 ... fit_first_half_neg_rho
%   5 ... fit_first_half_pos_rho
%   6 ... fit_first_half_k
%   7 ... fit_first_half_bound
%   8 ... fit_first_half_neg_rho
%   9 ... fit_first_half_pos_rho
%  10 ... RT learning index (end - begin)
bData = nan(numSubjects, 10);

% summary accuracy/RT data
uData = nan(numSubjects, 2, 3, 2); % accuracy/RT, low/high/all mu, first/second half

% For fits
% RT vs trial
numTrials = 800;
xax = (1:numTrials)';
fcn = @(b,x) b(1)+b(2).*exp(-x./b(3));
options = optimset('MaxFunEvals', 10000);

% Example sessions to plot
% exampleSessions = [46 11];
for ss = 1:numSubjects

    % get data
    sData = table2array(dataTable(strcmp(dataTable.subject, subjects{ss}), {'rho', 'mu', 'RT', 'correct'}));
    sData(:,2) = abs(sData(:,2)); % fold two directions

    % Collect mean-subtracted RTs (per mu) to fit
    trData = nan(size(sData,1),1); % mean-sub RT
    rs = nonanunique(sData(:,1));
    Lm = false(size(sData,1),2);
    for rr = 1:length(rs)
        Lr = sData(:,1) == rs(rr);
        mus = nonanunique(sData(Lr,2));
        for mm = 1:2
            Lm(Lr & sData(:,2)==mus(mm),mm) = true;
        end
    end
    for mm = 1:2
        trData(Lm(:,mm)) = sData(Lm(:,mm),3) - mean(sData(Lm(:,mm),3),'omitnan');
    end

    % Fit as a function of trial number, just for rho=0
    Lg = isfinite(trData) & sData(:,1)==0;
    xs = xax(Lg);
    ys = trData(Lg);
    [Bs,fval] = fminsearch(@(b) sum((ys-fcn(b,xs)).^2), [-0.5 1 30], options);

    % Plot example sessions
    %     if any(ss==exampleSessions)
    %
    %         % Accuracy
    %         subplot(5,2,find(ss==exampleSessions)); cla reset; hold on;
    %         plot(find(Lm(:,1)), nanrunmean(sData(Lm(:,1),4),25), 'k+')
    %         plot(find(Lm(:,2)), nanrunmean(sData(Lm(:,2),4),25), 'kx')
    %
    %         % RT
    %         subplot(5,2,2+find(ss==exampleSessions)); cla reset; hold on;
    %         plot(xax(Lg), trData(Lg), 'k.');
    %         plot(xax, fcn(Bs,xax), 'r-')
    %         title(sprintf('ss=%d', ss))
    %         axis([0 800 -4 12])
    %     end

    % Selection arrays for fits
    Lsub = strcmp(halfSessionTable.subject, subjects{ss});

    % Collect data
    bData(ss,:) = [ ...
        max(sData(:,1)), ...
        table2array(halfSessionTable(Lsub & LfirstHalf & Ldrift, 'value')), ...
        table2array(halfSessionTable(Lsub & LfirstHalf & Lbound, 'value')), ...
        table2array(halfSessionTable(Lsub & LfirstHalf & LrhoNeg, 'value')), ...
        table2array(halfSessionTable(Lsub & LfirstHalf & LrhoPos, 'value')), ...
        table2array(halfSessionTable(Lsub & ~LfirstHalf & Ldrift, 'value')), ...
        table2array(halfSessionTable(Lsub & ~LfirstHalf & Lbound, 'value')), ...
        table2array(halfSessionTable(Lsub & ~LfirstHalf & LrhoNeg, 'value')), ...
        table2array(halfSessionTable(Lsub & ~LfirstHalf & LrhoPos, 'value')), ...
        -diff(fcn(Bs,[2 numTrials]))];

    % Accuracy/RT summary
    Lh = cat(1, true(floor(size(Lm,1)/2), 1), false(ceil(size(Lm,1)/2), 1));
    Lh = [Lh ~Lh];
    Lm = cat(2, Lm, true(size(Lm,1),1));
    for hh = 1:2
        for mm = 1:3
            Lg = sData(:,4)>=0 & Lm(:,mm) & Lh(:,hh);
            uData(ss,1,mm,hh) = sum(sData(Lg,4)==1)./sum(Lg);
            uData(ss,2,mm,hh) = mean(sData(Lg,3), 'omitnan');
        end
    end
end

% Report stats
vars = {'Accuracy' , 'RT'};
for vv = 1:2
    vals = squeeze(uData(:,vv,3,:));
    disp(fprintf('Change in %s = %.2f±%.2f, p=%.3f\n', ...
        vars{vv}, ...
        mean(diff(vals,[],2)), sem(diff(vals,[],2)), ...
        signtest(vals(:,1), vals(:,2))))
end

% plot index vs k, B, rhos
rhos = unique(bData(:,1));
numRhos = length(rhos);
lms = [10 40; 0 4];
wh = 0.99.*ones(3,1);
for pp = 1:2
    for rr = 1:numRhos
        subplot(3,numRhos,(pp-1)*numRhos+rr); cla reset; hold on;
        Lr = bData(:,1) == rhos(rr);
        xs = bData(Lr,1+pp);
        ys = bData(Lr,5+pp);
        Lg = isfinite(xs) & isfinite(ys);
        plot(xs(Lg), ys(Lg), 'ko', 'MarkerFaceColor', wh)
        plot(lms(pp,:), lms(pp,:), 'k:')
        axis(lms(pp,[1 2 1 2]));            
        title(sprintf('rho=%.1f, p=%.2f', rhos(rr), signrank(xs(Lg), ys(Lg))))
    end
end

sy = {'d', 's'};
for rr = 1:numRhos
    subplot(3,numRhos,2*numRhos+rr); cla reset; hold on;
    plot([-1 1], [-1 1], 'k:')
    tstr = sprintf('rho=%.1f', rhos(rr));
    for ss = 1:2
        Lr = bData(:,1) == rhos(rr);
        xs = bData(Lr,3+ss);
        ys = bData(Lr,7+ss);
        Lg = isfinite(xs) & isfinite(ys);
        plot(xs(Lg), ys(Lg), cat(2, 'k', sy{ss}), 'MarkerFaceColor', wh)
        tstr = sprintf('%s, p=%.2f', tstr, signrank(xs(Lg), ys(Lg)));
    end
    axis([-1 1 -1 1]);
    title(tstr)
end
