function [flag,Path,slotPattern]=decode_user(Trans_cell_estimate,slotIdx,userIdx,SystemParameters,TransmissionPatternMatrix,parity)
n=SystemParameters.n;
l=SystemParameters.l;
B=SystemParameters.B;
L=SystemParameters.TotalSlots;

flag=0;
slotBits=Trans_cell_estimate{slotIdx};
Preamble=slotBits(userIdx,:);
if isempty(find(TransmissionPatternMatrix(:,1)==binaryVectorToDecimal(Preamble)))
    %keyboard
    slotPattern=zeros(1,n);
    slotPattern(1)=slotIdx;
    slotPattern(2:n)=slotIdx+sort(randperm(L-slotIdx,n-1),'ascend');
else
    slotPattern=TransmissionPatternMatrix(find(TransmissionPatternMatrix(:,1)==binaryVectorToDecimal(Preamble)),2:end);
    %assert(slotIdx==slotPattern(1));
    %keyboard
end


Path=[];
    new = [];
    %keyboard
    for k=1:size(Trans_cell_estimate{slotPattern(2)},1)
        index = parity_check(userIdx,slotIdx,slotPattern,k,Trans_cell_estimate,parity,SystemParameters);
        %keyboard
        if(index==1)
            new = [new;[userIdx k]];
            %keyboard
        end
    end
    
    prev=new;
    %keyboard;
    for i=3:n
        prev = new;
        new=[];
        for j=1:size(prev,1)
            for k=1:size(Trans_cell_estimate{slotPattern(i)},1)
                index = parity_check(prev(j,:),slotIdx,slotPattern,k,Trans_cell_estimate,parity,SystemParameters);
                if(index==1)
                    new = [new;[prev(j,:) k]];
                    %keyboard
                end
                
            end
            
        end
    end
    if isempty(new)
        %keyboard
    end
    if size(new,1)==1
        flag=1;
        Path=new;
    end
    if size(new,1)>1
        flag=1;
        [flag1, Path]=isSameMessage(new,slotPattern,Trans_cell_estimate);
        %keyboard
    end
end