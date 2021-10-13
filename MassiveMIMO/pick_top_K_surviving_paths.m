function survivingPathsTopK = pick_top_K_surviving_paths(survivingPathsNonDup,encoded_tx_soft_estimate,systemParameters)

L=systemParameters.L;
listSize=systemParameters.listSize;
Ka=systemParameters.Ka;

Paths=[];
for i=1:listSize
    temp = survivingPathsNonDup{i,L};
    for j=1:size(temp,1)
        Paths = [Paths;temp(j,:)];
    end
end
survivingPathsTopK = Paths;

if size(Paths,1) > Ka
    c=zeros(size(Paths,1),L);
    for j=1:size(Paths,1)
        for k=1:L
            B=encoded_tx_soft_estimate{k};
            c(j,k)=B(Paths(j,k));
        end
    end
    d=sum(c,2);
    
    [~,dSortedIndices]=sort(d);
    survivingPathsTopK = Paths(dSortedIndices(end-Ka+1:end),:);
    
end

end