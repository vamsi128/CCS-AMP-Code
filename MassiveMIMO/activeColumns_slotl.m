function activeColumns = activeColumns_slotl(permissibleParityBits,l,systemParameters)
messageLengths=systemParameters.messageLengths;
if messageLengths(l)~=0
    % Binary indices for active columns consist of all possible message bits concatenated with
    % permissible parity bits
    
    indices = [1:2^(messageLengths(l))]';
    temp1=dec2bin(indices-1,messageLengths(l));
    temp2=char(num2cell(temp1));
    temp3 = reshape(str2num(temp2),[],messageLengths(l));
   
    activeIndicesBinary=[temp3 kron(permissibleParityBits,ones(2^messageLengths(l),1))];
else
    activeIndicesBinary=permissibleParityBits;
end

activeColumns=bin2dec(num2str(activeIndicesBinary))+1;
%activeColumns=binaryVectorToDecimal(activeIndicesBinary)+1;
end