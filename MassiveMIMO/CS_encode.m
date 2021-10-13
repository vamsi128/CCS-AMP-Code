function transmitted_signals = CS_encode(encoded_tx_messages,SensingMatrix,systemParameters)

L = systemParameters.L;
% transmitted_signals{l} corresponds to the signals transmitted by 
% active users during slot l
transmitted_signals = cell(1,L);

for l=1:L
    % Decimal representation of encoded messages in each section
    % Add 1 to avoid 0 indexing
    encoded_tx_messages_decimal = bin2dec(num2str(encoded_tx_messages{l}))+1;
    % Signals transmitted by actve users during slot l
    transmitted_signals{l} = SensingMatrix(:,encoded_tx_messages_decimal);
end

end