clear;
close all;
clc;

%% Input parameters
KaVector = [100];
%systemParameters.Ka=25;   % Number of active users
systemParameters.M = 50;  % Number of antennas at BS
systemParameters.L=32;    % Number of sections
systemParameters.J = 12;  % Length of each section
systemParameters.parityLengths=[0 9*ones(1,28) 12 12 12]; % Parity bits in each section
systemParameters.totalParityLength=sum(systemParameters.parityLengths);  % Total number of parity bits
systemParameters.totalMsgLength=systemParameters.L*systemParameters.J-systemParameters.totalParityLength; % Total number of message bits
systemParameters.messageLengths=systemParameters.J*ones(1,systemParameters.L)-systemParameters.parityLengths; % Message bits in each section
systemParameters.num_channel_uses=3200; % Total number of channel uses
systemParameters.num_channel_uses_section=systemParameters.num_channel_uses/systemParameters.L; % Channel uses per section
CSParameters.delta=0;
CSParameters.eCCS = 1; % Set this flag to 1 to enable eCCS, 0 for legacy CCS % Legacy CCS is Fengler's scheme and eCCS is our ASILOMAR scheme
CSParameters.numIterCov = 6; % Number of iterations in covariance based detection algorithm (inner decoder)
max_sim = 100;  
EbN0dB=0; % Operating Eb/N0
SensingMatrix=randn(systemParameters.num_channel_uses_section, 2^systemParameters.J)+1i*randn(systemParameters.num_channel_uses_section, 2^systemParameters.J); % Inner codebook

% Normalize columns to L_2 norm square = num_channel_uses_section
for i=1:2^systemParameters.J
    SensingMatrix(:,i) = (SensingMatrix(:,i)/norm(SensingMatrix(:,i)))*sqrt(systemParameters.num_channel_uses_section);
end

for KaIndex = 1:length(KaVector)
    systemParameters.Ka = KaVector(KaIndex);
    systemParameters.listSize=systemParameters.Ka+CSParameters.delta; % List size output by the inner decoder
    error = zeros(1,length(EbN0dB));
    A1 = sprintf('Results.txt');
    for s=1:length(EbN0dB)
        tstart = tic;
        for sims = 1:max_sim
            
            % Generate active user message sequences
            tx_messages = randi([0,1],systemParameters.Ka,systemParameters.totalMsgLength);
            
            % Create Rademacher generator matrices for the outer tree code
            G = createG(systemParameters);
            
            % Outer-encode the message sequences
            encoded_tx_messages = tree_encode(tx_messages,G,systemParameters);
            
            %% Inner/CS encoding
            transmitted_signals = CS_encode(encoded_tx_messages,SensingMatrix,systemParameters);
            
            %% MAC channel
            [received_signalMatrix,sigma] = MAC_SIMOchannel(transmitted_signals, EbN0dB(s), systemParameters);
            
            %% Tree/adaptive CS decoding starts here
            survivingPaths = cell(systemParameters.listSize,systemParameters.L); % Keeps track of surviving paths in the tree
            encoded_tx_messages_estimate = cell(1,systemParameters.L);  % output messages of inner decoders
            encoded_tx_soft_estimate = cell(1,systemParameters.L);    % Soft outputs of inner decoders
            
            activeColumns = cell(1,systemParameters.L); % Keeps track of active columns in sensing matrix at every slot
            
            for j=1:systemParameters.listSize
                survivingPaths{j,1}=j;   % Root nodes of all trees
            end
            
            for l=1:systemParameters.L
                %disp(l);
                if l==1
                    % All columns in sensing matrix are active during slot 1
                    activeColumns{l} = 1:2^systemParameters.J;
                else
                    if CSParameters.eCCS==0
                        % Legacy CCS->all columns are active
                        activeColumns{l} = 1:2^systemParameters.J;
                    else
                        % Determine active columns based on surviving paths in tree
                        activeColumns{l} = find_active_columns(encoded_tx_messages_estimate,survivingPaths,l,systemParameters,G);
                    end
                end
                
                % Restrict sensing matrix to columns corr to activeColumns
                updatedSensingMatrix=SensingMatrix(:,activeColumns{l});
                
                % Inner decoder
                [encoded_tx_messages_estimate{l}, encoded_tx_soft_estimate{l}] = Cov_decoder_slot(received_signalMatrix{l},updatedSensingMatrix,systemParameters,CSParameters,activeColumns{l},sigma,encoded_tx_messages{l});

                %command below is only to test outer decoder
                %encoded_tx_messages_estimate{l} = encoded_tx_messages{l};
               
            
                                
                %%
                % Update survivingPaths through tree decoder
                if l>=2
                    for j=1:systemParameters.listSize
                        Paths=survivingPaths{j,l-1}; % SurvivingPaths corr to root node j until slot l-1
                        if ~isempty(Paths)
                            for k=1:size(Paths,1)
                                path=Paths(k,:); % kth surviving path corr to root node j until slot l-1
                                % Link path with newly decoded candidates
                                survivingPaths=tree_decoder_update_suriving_paths(survivingPaths,path,l,j,encoded_tx_messages_estimate,systemParameters,G);
                            end
                        end
                        
                    end
                end
            end
            
            % Remove duplicate paths
            survivingPathsNonDup = remove_dup_paths(survivingPaths,encoded_tx_messages_estimate,systemParameters); 
            
            % If the number of surviving paths is more than Ka, pick top Ka
            % of them based on soft info provided by inner decoder
            survivingPathsTopK = pick_top_K_surviving_paths(survivingPathsNonDup,encoded_tx_soft_estimate,systemParameters);
            
            % Check
            assert(size(survivingPathsTopK,1) <= systemParameters.Ka);
           
            % Extact message bits from surviving paths
            decoded_tx_messages=extract_msg_bits(survivingPathsTopK,encoded_tx_messages_estimate,systemParameters);
            
            % Computation of PUPE
            if isempty(decoded_tx_messages)
                error(s) = error(s) + size(tx_messages,1);
            else
                for kk=1:size(tx_messages,1)
                    if ~ismember(tx_messages(kk,:),decoded_tx_messages,'rows')
                        error(s)=error(s)+1;
                    end
                end
            end
            
            disp(sims)
            %disp(error(s))
 
        end
        timeElapsed = toc(tstart);
        
        disp(error/systemParameters.Ka/max_sim)
        f1 = fopen(A1,'at');
        fprintf(f1,'Ka=%d\n',systemParameters.Ka);
        fprintf(f1,'M=%d\n',systemParameters.M);
        fprintf(f1,'eCCS=%d\n',CSParameters.eCCS);
        fprintf(f1,'errorRate=%f\n',error(s)/systemParameters.Ka/max_sim);
        fprintf(f1,'timeElapsed=%f\n\n',timeElapsed/max_sim);
        
        fclose(f1);
    end
end