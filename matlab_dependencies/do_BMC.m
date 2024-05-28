function [posterior,out] = do_BMC(lle,subjvar,modelvar,fitvar)
%compute exceedance pr and protected excedance pr based on AICs.
if ~exist('subjvar','var') || isempty(subjvar)
    subjvar = 'subject';
end

if ~exist('modelvar','var') || isempty(modelvar)
    modelvar = 'model';
end

if ~exist('fitvar','var') || isempty(fitvar)
    fitvar = 'aic';
end

lle_VBA = lle(:,{subjvar,modelvar,fitvar});
%Stephan et al 2009 AIC is normal AIC/-2 ie log(lle) - k; this is what
%the BMC function expects (I have a question out asking why)
lle_VBA.(fitvar) = lle_VBA.(fitvar)./-2;
lle_VBA = unstack(lle_VBA,fitvar,modelvar);

opts.MaxIter = 1e4;
opts.modelNames = lle_VBA.Properties.VariableNames(2:end);

[posterior,out] = VBA_groupBMC(lle_VBA{:,2:end}',opts);
%compute protected EP: Rigoux et al., 2014. Code from:
%http://mbb-team.github.io/VBA-toolbox/wiki/BMS-for-group-studies/
out.PEP = (1-out.bor)*out.ep + out.bor/length(out.ep);

end