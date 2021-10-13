function survivingPathsNonDup = remove_dup_paths(survivingPaths,encoded_tx_messages_estimate,systemParameters)
L=systemParameters.L;
listSize=systemParameters.listSize;

survivingPathsNonDup = survivingPaths;

for i=1:listSize
    A=survivingPaths{i,L};
    m=size(A,1);
    P=[];
    if m>0
        if m>=2
            [flag, path]=isSameMessage(A,encoded_tx_messages_estimate);
            
            if flag
                P=path;
            else
                %r=randi([1,m],1);
                P=path;
                %keyboard
            end
            %keyboard
        else
            P=A;
            %keyboard
        end
    end
    survivingPathsNonDup{i,L} = P;
    
end



end






