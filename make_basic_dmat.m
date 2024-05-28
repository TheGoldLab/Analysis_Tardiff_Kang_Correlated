function dat2Fit=make_basic_dmat(data,SNRvar,choicevar)
    if ~exist('SNRvar','var')
        SNRvar = 'SNR';
    end
    if ~exist('choicevar','var')
        choicevar = 'response';
    end

    dummy = ones(height(data), 1);
    dat2Fit = [dummy data.(SNRvar) data.(choicevar)]; %add choice data
end
