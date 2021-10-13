function [out parity] = create_msg_plus_parity(msg,SP)
total_parity=SP.total_parity;
totalMsg=SP.totalMsg;
B=SP.B;
n=SP.n;
l=SP.l;
m=[B B*ones(1,n-1)-l];
parity = cell(n,1);
message_plus_parity=msg(:,1:m(1));
%%
for i=2:n
    parity{i}=randi([0,1],l(i-1),(sum(m(1:i-1))));
    parity_bits=mod(msg(:,1:sum(m(1:i-1)))*parity{i}',2);
    message_plus_parity=[message_plus_parity msg(:,sum(m(1:i-1))+1:sum(m(1:i))) parity_bits];
end
out = message_plus_parity;
end