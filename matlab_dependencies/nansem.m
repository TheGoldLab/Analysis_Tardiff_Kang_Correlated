function [se]=nansem(x)
    se = nanstd(x)./sqrt(sum(~isnan(x)));
end