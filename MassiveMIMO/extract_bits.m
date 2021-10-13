function Decoded_msg_bits=extract_msg_bits(survivingPaths,Trans_cell_estimate,systemParameters)

n=systemParameters.n;
listSize=systemParameters.listSize;
mL=SystemParameters.messageLengths;
extracted_bits=[];


for i=1:listSize
    A=survivingPaths{i,n};
    m=size(A,1);
    
    for j=1:m
        log_bits=[];
        P=A(j,:);
        for k=1:n
            %keyboard
            block=Trans_cell_estimate{i};
            if isempty(block)
                keyboard
            end
            %keyboard
            log_bits=[log_bits block(P(i),1:mL(k))];
        end
        extracted_bits=[extracted_bits;log_bits];
    end
    
    
    
end







end