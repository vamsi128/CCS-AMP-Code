function s=populate_lookUp(upcomingParity,i,systemParameters)
m=systemParameters.messageLengths;
if m(i)~=0
    %keyboard
    indicesBinary=[decimalToBinaryVector([0:2^(m(i))-1]',m(i)) kron(upcomingParity,ones(2^m(i),1))];
else
    indicesBinary=upcomingParity;
end
%keyboard
s=binaryVectorToDecimal(indicesBinary)+1;
end