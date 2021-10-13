function [T_est,H_estimate]=CS_decode_slot(Received_signal_cell,effectiveSensingMatrix,systemParameters,CSParameters,lookUp,SNR)
listSize=systemParameters.listSize;
Ka=systemParameters.Ka;
B=systemParameters.B;
m=CSParameters.maxSens;
totalMsg=systemParameters.totalMsg;
n=systemParameters.n;
num_channel_uses_per_slot=systemParameters.num_channel_uses_per_slot;
%P=(2*totalMsg*10.^(SNR/10))/(n*num_channel_uses_per_slot);


C= (effectiveSensingMatrix-m)/2;

Y=Received_signal_cell;


Y=0.5*(-Ka*m*ones(length(Y),1)+Y);
%keyboard
%tic
Xhat=lsqnonneg(C,Y);
%fprintf('%f secs\n',toc)
%keyboad
[sortedXhat,sortIndices]=sort(Xhat);

%keyboard
if length(sortedXhat)>=listSize
    Xhat_list=sortedXhat(end-listSize+1:end)';
    list=sortIndices(end-listSize+1:end);
else
    Xhat_list=sortedXhat';
    list=sortIndices;
    
end
%keyboard
indices=lookUp(list);
Xhat_binary=decimalToBinaryVector(indices-1,B);

%keyboard
T_est=Xhat_binary;

H_estimate=Xhat_list;
end