function pathMsgBits=fing_path_msg_bits(path,encoded_tx_messages_estimate,SystemParameters)
pathMsgBits=[];
messageLengths=SystemParameters.messageLengths;

for i=1:length(path)
    A=encoded_tx_messages_estimate{i}; % Encoded message estimates of ith slot
    pathMsgBits=[pathMsgBits A(path(i),1:messageLengths(i))]; % Estimated Message bits corr to ith slot
end

end