% Cov matching

clear all
clc
close all
K = 50;                            % Number of users
L = 100;                           % Length of the spreading sequence
Nc = 300;                          % Length of Polar codeword
B = 100;                           % Number of uncoded bits
N = 30000;                         % Total no. of channel uses

Kmax = 2^12;                       % Max no of spreading sequences
EbN0dB = [-4,-2,0,2,4,6,8];
PmissDet = zeros(1,length(EbN0dB)); % Prob. of missed detection
sigma_n = 1;  % std of noise
maxSim = 10;

numIterCov = 10; % Number of iteration rounds for covariance matching

ML=1;             % Set this to 1 for ML and 0 for nnLS

for s = 1:length(EbN0dB)
    P = 2*B*sigma_n^2*10^(0.1*EbN0dB(s))/N; % Power of coded symbol
    
    for sim = 1:maxSim
        
        activeColumns = randperm(Kmax,K); % Active columns/users
        
        A = randn(L,Kmax); % Signature matrix, Normalize columns if required
        
        gamma = zeros(1,Kmax);
        gamma(activeColumns) = P;   % gamma(k)=P if user k is active
        Gamma = diag(gamma); % Activity pattern matrix
        
        % C(k,:) is the modulated Polar codeword transmitted by user k \in [1:Kmax]
        % Replace this with Polar codewords 
        C = 2*randi([0 1],Kmax,Nc)-1; 
        
        Z = sigma_n*randn(L,Nc); % Noise
        
        % Received signal in a form reminiscent of eqn (4) in https://arxiv.org/pdf/1910.11266.pdf
        % lth row of Y is the received signal at BS at slot l in [1:L]
        % so, in your code reshape the received signal Y as an (LxNc)
        % matrix and that will be the equivalent of Y in next line
        Y = A*sqrt(Gamma)*C + Z; 
        
        empiricalCovMatrix = (1/Nc)*(Y*Y'); % Empirical covariance matrix
        
        [gamma_hat, normevolution] = activityDetect(empiricalCovMatrix, A, Kmax, Nc, P, sigma_n, numIterCov, ML, gamma);
        
        [~,activeColumns_estimate] = maxk(gamma_hat,K+10); % Estimate of active users
        
        
        PmissDet(s) = PmissDet(s) + K - length(intersect(activeColumns,activeColumns_estimate));
        
        
    end
    PmissDet(s) = PmissDet(s)/maxSim;
end
PmissDet