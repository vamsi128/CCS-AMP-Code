function [lookUp]=find_effective_sensing_matrix(Trans_cell_estimate,survivingPaths,i,SystemParameters,parity)
lookUp=[];
listSize=SystemParameters.listSize;
l=SystemParameters.l;
n=SystemParameters.n;
B=SystemParameters.B;
m=SystemParameters.messageLengths;
for j=1:listSize
    A=survivingPaths{j,i-1};
    for k=1:size(A,1)
        path=A(k,:);
        pathMsgBits=fing_path_msg_bits(path,Trans_cell_estimate,SystemParameters);
        upcomingParity=find_upcoming_parity_bits(pathMsgBits,parity{i});
        lookUp=[lookUp populate_lookUp(upcomingParity,i,SystemParameters)];
    end
end
%keyboard
lookUp=unique(lookUp);
end