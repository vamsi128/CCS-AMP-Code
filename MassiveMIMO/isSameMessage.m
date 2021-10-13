function [flag, Path]=isSameMessage(new,encoded_tx_messages_estimate)
flag=0;
Path=[];
DecBits=[];
n=size(new,2);
%keyboard
for i=1:size(new,1)
    s=new(i,:);
    B=[];
    for j=1:n
        A=encoded_tx_messages_estimate{j};
        %keyboard
        B=[B A(s(j),:)];
    end
    DecBits(i,:)=B;
end
K=kron(DecBits(1,:),ones(size(new,1),1));
sm=sum(abs(DecBits-K));
%keyboard
if sm==0
    flag=1;
    Path=new(1,:);
    %keyboard
else
    Path=new;
    %keyboard
end


%keyboard
end