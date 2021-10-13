function index = parity_check(computedParityBits,k,l,encoded_tx_messages_estimate,systemParameters)

J = systemParameters.J;
parityLengths = systemParameters.parityLengths;

currentBlock = encoded_tx_messages_estimate{l};
ParityBits = currentBlock(k,J-parityLengths(l)+1:J);

if(sum(abs(computedParityBits-ParityBits))==0)
    index=1;
else
    index=0;
end

end