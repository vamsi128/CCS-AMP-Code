clear all;
J= 1024;
N=2^14;
SensingMatrix=zeros(J,N);
Index = randperm(N,J);
%Index=1:J;
for i=1:J
    for j=1:N
        SensingMatrix(i,j)=exp(1i*2*pi*(Index(i)-1)*(j-1)/N);
    end
end
SensingMatrix=sqrt(2)*[real(SensingMatrix);imag(SensingMatrix)];
save('Four_2048cross2pow14','SensingMatrix');
%keyboard
