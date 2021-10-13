function index = parity_check_prime(path,k,Trans_cell_estimate,parity,SystemParameters)
%keyboard
n=SystemParameters.n;
B=SystemParameters.B;
l=SystemParameters.l;
Lpath = length(path);
msg = [];
total_parity=sum(l);
totalMsg=n*B-total_parity;
m=[B B*ones(1,n-1)-l];
%keyboard

for i=1:Lpath
    block=Trans_cell_estimate{i};
    msg = [msg block(path(i),1:m(i))];   
end
%keyboard
block=Trans_cell_estimate{Lpath+1};
    sg =  block(k,:);
   %keyboard
    parity_bits = sg(B - l(Lpath)+1:B);
    %keyboard
    compute_parity = mod(msg*parity{Lpath+1}',2);
    %keyboard
    if(sum(abs(compute_parity-parity_bits))==0)
        index=1;
    else
        index=0;
    end
end