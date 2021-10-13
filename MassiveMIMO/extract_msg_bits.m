function decoded_tx_messages=extract_msg_bits(survivingPathsTopK,encoded_tx_messages_estimate,systemParameters)
L=systemParameters.L;
listSize=systemParameters.listSize;
messageLengths=systemParameters.messageLengths;
decoded_tx_messages=[];

for r=1:size(survivingPathsTopK,1)
    log_bits=[];
    Path = survivingPathsTopK(r,:);
    for k=1:L

        block=encoded_tx_messages_estimate{k};
        
        log_bits=[log_bits block(Path(k),1:messageLengths(k))];
    end
    decoded_tx_messages=[decoded_tx_messages;log_bits];
end


end






