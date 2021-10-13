function [per CmprsSnsgOutBits] = create_cmpsSngOutBits(msg_cell,n,B,Ka);

CmprsSnsgOutBits = zeros(Ka,n*B);

for i=1:n
    perm(i,:) = randperm(Ka);
    CmprsSnsgOutBits(:,B*(i-1)+1:B*i) = msg_cell(perm(i,:),B*(i-1)+1:B*i);
    
end
per=perm;
end