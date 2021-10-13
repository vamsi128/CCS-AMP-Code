function [Trans_cell_estimate,H_estimate]=CS_decode_slot(Received_signal_cell,effectiveSensingMatrix,SystemParameters,CSParameters,lookUp)
listSize=SystemParameters.listSize;
delta=CSParameters.delta;
thresh1=CSParameters.twocollisionThreshold;
thresh2=CSParameters.threecollisionThreshold;
m=CSParameters.maxSens;
C= (SensingMatrix-m)/2;
for i=1:size(Received_signal_cell,2)
    Y=Received_signal_cell{i};
    %K_estimate{i}=estimateK(Y(1:num_channel_uses_activeuser_estimate));
    K_estimate{i}=size(Trans_cell{i},1);
    listSize{i}=K_estimate{i}+delta;
    %keyboard
    Y=0.5*(-K_estimate{i}*m*ones(length(Y),1)+Y);
    
    %keyboard
    tic
    Xhat=lsqnonneg(C,Y);
    fprintf('%f secs\n',toc)
    %keyboad
    [sortedXhat,sortIndices]=sort(Xhat);
    list=sortIndices(end-listSize{i}+1:end);
    %keyboard
    Xhat_list=sortedXhat(end-listSize{i}+1:end)';
    %keyboard
    Xhat_binary=decimalToBinaryVector(list-1,B);
    Xhat_binary_collisionResolved=collisionResolve(Xhat_binary,Xhat_list,thresh1,thresh2);
    %keyboard
    Trans_cell_estimate{i}=Xhat_binary_collisionResolved;
    listSize{i}=size(Trans_cell_estimate{i},1);
    H_estimate{i}=Xhat_list;
end
end