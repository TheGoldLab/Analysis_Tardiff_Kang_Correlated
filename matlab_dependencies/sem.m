function [se]=sem(x)
    se = std(x)/sqrt(length(x));
end