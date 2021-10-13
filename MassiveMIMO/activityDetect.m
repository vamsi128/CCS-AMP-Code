function [gamma_t,normevolution] = activityDetect(empiricalCovMatrix, A, Kmax, Nc, P, sigma_n, numIterCov, ML, gamma)
L = size(empiricalCovMatrix,1);
normevolution = zeros(1,numIterCov);
% Initialization
covInv_t = (1/sigma_n^2)*eye(L);
gamma_t = zeros(1,Kmax);

for n=1:numIterCov
    for k=1:Kmax
        
        if ML
            temp = (A(:,k)'*covInv_t*empiricalCovMatrix*covInv_t*A(:,k)-A(:,k)'*covInv_t*A(:,k))./((A(:,k)'*covInv_t*A(:,k))^2);         
        else
            temp = (A(:,k)'*(empiricalCovMatrix-inv(covInv_t))*A(:,k))/(norm(A(:,k))^4);
        end
        d0star = max(temp,-gamma_t(k));
        dstar = d0star;

        %dstar = min(d0star,1-gamma_t(k)); % Fix me
        
        
        gamma_t(k) = gamma_t(k) + dstar;
        
        covInv_t = covInv_t - (dstar*covInv_t*A(:,k)*A(:,k)'*covInv_t)/(1 + dstar*A(:,k)'*covInv_t*A(:,k));
    end
    normevolution(n)= norm(gamma-gamma_t)/norm(gamma);
end

end