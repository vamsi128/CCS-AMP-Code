function survivingPathsprime=finalize_surviving_paths(survivingPaths,encoded_tx_soft_estimate,systemParameters)
listSize=systemParameters.listSize;
L=systemParameters.L;
survivingPathsprime=cell(listSize,1);
for i=1:listSize
    A=survivingPaths{i,L};  
    if size(A,1) <=1
        survivingPathsprime{i}=A;
    else
        c=zeros(size(A,1),L);
        for j=1:size(A,1)
            for k=1:L
                B=encoded_tx_soft_estimate{k};
                c(j,k)=B(A(j,k));
            end
        end
        d=sum(c,2);
        [~,maxIdx]=max(d);
        survivingPathsprime{i}=A(maxIdx,:);
    end
end


end