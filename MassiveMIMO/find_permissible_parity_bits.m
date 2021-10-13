function permissibleParityBits=find_permissible_parity_bits(pathMsgBits,G)
permissibleParityBits=mod(pathMsgBits*G,2);
end