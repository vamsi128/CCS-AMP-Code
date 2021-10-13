function [received_signalMatrix,sigma] = MAC_SIMOchannel(transmitted_signals, EbN0dB, systemParameters)

num_channel_uses = systemParameters.num_channel_uses;
num_channel_uses_section = systemParameters.num_channel_uses_section;
L = systemParameters.L;
totalMsgLength = systemParameters.totalMsgLength;
% Standard deviation of noise
sigma = sqrt((num_channel_uses/(totalMsgLength))*10^(-EbN0dB/10)); % Check this, should there be 2 in denominator?

received_signalMatrix = cell(1,L);

for l=1:L
    % MIMO channel matrix corr to slot l
    H = (1/sqrt(2))*(randn(systemParameters.Ka,systemParameters.M)+1i*randn(systemParameters.Ka,systemParameters.M));
    noise = (sigma/sqrt(2))*(randn(num_channel_uses_section,systemParameters.M)+1i*randn(num_channel_uses_section,systemParameters.M));
    % received_signal{l} is signal received at access point during slot l
    %noise=0;
    received_signalMatrix{l} = transmitted_signals{l}*H + noise;
end

end