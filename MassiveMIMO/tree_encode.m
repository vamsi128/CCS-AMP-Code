function encoded_tx_messages = tree_encode(tx_messages,G,systemParameters)

L = systemParameters.L;
messageLengths = systemParameters.messageLengths;
% encoded_tx_messages{l} consists of encoded messages corresponding to
% section l of all active users
encoded_tx_messages = cell(1,L);
encoded_tx_messages{1} = tx_messages(:,1:messageLengths(1));
%%
for l=2:L
    % Compute parity bits for section l
    parity_bits=mod(tx_messages(:,1:sum(messageLengths(1:l-1)))*G{l},2);
    % Append parity bits to message bits for section l; systematic encoding
    encoded_tx_messages{l} = [tx_messages(:,sum(messageLengths(1:l-1))+1:sum(messageLengths(1:l))) parity_bits];
end

end