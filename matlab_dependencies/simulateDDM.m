function [pmfs_, cmfs_, gMeans_] = simulateDDM(bound, snrScale, rScale, ...
    startBias, meBias, options)
% Simulate DDM with two (potentially) correlated observations per frame
%   snrScale scales the generative mean (i.e., changes the SNR)
%   rScale scales the paired evidence by the paired correlation: 
%       1/(1-rScale*r)
arguments
    bound=0.2;
    snrScale=1;
    rScale=1;
    startBias=0;
    meBias=0;
    options.llrs=logspace(-1,1,5)';
    options.rs=[-0.6 0 0.6];
    options.gSigma=0.1;
    options.nTrials=10000;
    options.maxFramesPerTrial=1000;
end

% Calculate generative means per (expected) LLR (rows) and r (columns),
%   for the given generative std
% Expected LLR per pair = (2 * mu) (2 * mu) / (2 * sigma^2 * (1 + r))
% thus mu = sigma * sqrt(0.5 * LLR * (1+r))
[LLRGrid,RGrid] = meshgrid(options.llrs, options.rs);
gMeans_ = options.gSigma.*sqrt(0.5 * LLRGrid'.*(1+RGrid'));

% Simulate separately per r
gVar = options.gSigma^2;
sigma = eye(2).*gVar;
covs = rScale.*options.rs.*gVar;
nTr  = options.nTrials*options.maxFramesPerTrial;
pmfs_ = nans(length(options.llrs), length(options.rs));
cmfs_ = nans(length(options.llrs), length(options.rs), 2);
tdat = nans(1, options.nTrials);
if rScale == -1
    scale = ones(size(gMeans_));
else
    scale = gMeans_./repmat((gVar.*(1+rScale.*options.rs)),size(gMeans_,1),1);
end

for rr = 1:length(options.rs)
    
    % Get matrix of momentary evidence
    sigma([2 3]) = covs(rr);
    
    % For each LLR
    for ll = 1:length(options.llrs)
        
        % Get samples
        R = mvnrnd(gMeans_(ll, [rr rr]).*snrScale, sigma, nTr);
        
        % Compute DV from scaled sum
        % NOTE: true LLR is 2.*mu.*(x1 + x2)./(sigma^2 * (1+rho))
        L = scale(ll,rr).*sum(R,2) + meBias;
        DV = startBias/bound + cumsum(reshape(L, ...
            [options.maxFramesPerTrial options.nTrials]));
                
        % Loop through the frames
        tdat(:) = nan;
        for ff = 1:options.maxFramesPerTrial
            Lcor = ~isfinite(tdat) & (DV(ff,:) >= bound);
            tdat(Lcor) = ff;
            Lerr = ~isfinite(tdat) & (DV(ff,:) <= -bound);
            tdat(Lerr) = -ff;
            if all(isfinite(tdat))
                break;
            end
        end
        if any(~isfinite(tdat))
            fprintf('WARNING: %d nc trials\n', sum(~isfinite(tdat)))
        end
        pmfs_(ll,rr) = sum(tdat>0)./sum(isfinite(tdat)).*100;
        cmfs_(ll,rr, 1) = mean(tdat(tdat>0));
        cmfs_(ll,rr, 2) = -mean(tdat(tdat<0));

        % Report correlation between empirical
        %         vars = nan(options.nTrials, 1);
        %         mer = reshape(L,[options.maxFramesPerTrial options.nTrials]);
        %         Lcor = tdat > 0;
        %         for tt = find(Lcor)
        %             vars(tt) = var(mer(1:ff,tt));
        %         end
        %         disp([options.rs(rr) options.llrs(ll) corr(vars(Lcor), tdat(Lcor)')])
    end
end
