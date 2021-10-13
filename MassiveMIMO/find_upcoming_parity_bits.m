function upcomingParityBits=find_upcoming_parity_bits(pathMsgBits,parity)
upcomingParityBits=mod(pathMsgBits*parity',2);
end