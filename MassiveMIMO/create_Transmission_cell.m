function C=create_Transmission_cell(M,T,SystemParameters)
L=SystemParameters.TotalSlots;
B=SystemParameters.B;
C=cell(1,L);
Ka=size(M,1);
for i=1:Ka
    Preamble=binaryVectorToDecimal(M(i,1:B));
    TransPattern=T(find(T(:,1)==Preamble),2:end);
    for j=1:length(TransPattern)
        C{TransPattern(j)}=[C{TransPattern(j)};M(i,(j-1)*B+1:(j-1)*B+B)];
    end
end
end