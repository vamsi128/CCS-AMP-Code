function [encoded_tx_messages_estimate, encoded_tx_soft_estimate] =Cov_decoder_slot(received_signalMatrix,updatedSensingMatrix,systemParameters,CSParameters,activeColumns,sigma,encoded_tx_messages)

listSize=systemParameters.listSize;
Ka=systemParameters.Ka;
J=systemParameters.J;
totalMsgLength=systemParameters.totalMsgLength;
L=systemParameters.L;
M = systemParameters.M;
num_channel_uses_section=systemParameters.num_channel_uses_section;
numIterCov = CSParameters.numIterCov;
Kmax = size(updatedSensingMatrix,2); % Number of columns of updated sensing matrix

empiricalCovMatrix = (1/M)*(received_signalMatrix*received_signalMatrix'); % Compute empirical covariance matrix of received signal

normevolution = zeros(1,numIterCov);  % check

% Initialization
covInv_t = (1/sigma^2)*eye(num_channel_uses_section);
gamma_t = zeros(1,Kmax);
ML=1; % Set this to one for ML decoder and 0 for nnLS decoder


% Below is just a acheck
%encoded_tx_messages_decimal = bin2dec(num2str(encoded_tx_messages))+1;
%gammaTrue = zeros(1,Kmax);
%gammaTrue(encoded_tx_messages_decimal)=1;

%% Covariance based detection algorithm.
% See https://arxiv.org/pdf/1901.00828.pdf
for n=1:numIterCov
    for k=1:Kmax
        
        if ML
            temp = (updatedSensingMatrix(:,k)'*covInv_t*empiricalCovMatrix*covInv_t*updatedSensingMatrix(:,k)-updatedSensingMatrix(:,k)'*covInv_t*updatedSensingMatrix(:,k))./((updatedSensingMatrix(:,k)'*covInv_t*updatedSensingMatrix(:,k))^2);         
        else
            temp = (updatedSensingMatrix(:,k)'*(empiricalCovMatrix-inv(covInv_t))*updatedSensingMatrix(:,k))/(norm(updatedSensingMatrix(:,k))^4);
        end
        %keyboard
        d0star = max(real(temp),-gamma_t(k));
        dstar = d0star;

        %dstar = min(d0star,1-gamma_t(k)); %
        
        % Update state vector
        gamma_t(k) = gamma_t(k) + dstar;
        
        % Covariance matrix update
        covInv_t = covInv_t - (dstar*covInv_t*updatedSensingMatrix(:,k)*updatedSensingMatrix(:,k)'*covInv_t)/(1 + dstar*updatedSensingMatrix(:,k)'*covInv_t*updatedSensingMatrix(:,k));
    end
 
    %Track L2 norm of error 
    %normevolution(n)= norm(gammaTrue-gamma_t)/norm(gammaTrue);
end


% Sort the indices
[sortedXhat,sortIndices]=sort(gamma_t);

if length(sortedXhat)>=listSize
    Xhat_list=sortedXhat(end-listSize+1:end)';
    list=sortIndices(end-listSize+1:end);
else
    Xhat_list=sortedXhat';
    list=sortIndices;    
end

% Transform into correct indices
indices=activeColumns(list);
temp1=dec2bin(indices-1,J);
temp2=char(num2cell(temp1));

encoded_tx_messages_estimate=reshape(str2num(temp2),[],J);
encoded_tx_soft_estimate=Xhat_list;
end