% tree_encoder  Encodes a message into patches using parity check digits
%
% input_bits      B x K matrix of the K B-bit messages
% N               number of bits per patch
% n               number of patches
% l               length-n vector of number of parity check digits
%                 recommend: n=1: l=0, n=2: l=[0 15], n=4: l = [0 10 10 15]
%
% patch_bits       N x K x n tensor of the K N-bit messages in each patch
%
% Regardless of input, zero parity check digits are used for the first
% patch, and a number of parity check digits is used for the last patch to
% agree with the total number of bits B. Ensure that B + sum(l) = N x n.
%
% Code is based on 'A Coupled Compressive Sensing Scheme for Unsourced 
% Multiple Access' by Amalladinne et al. 2018 (arXiv 1806.00138)

function [patch_bits parity] = tree_encoder(input_bits,N,n,l)

l(1) = 0;
l(n) = N*n - size(input_bits,1) - sum(l) + l(n);

patch_bits(:,:,1) = input_bits(1:N,:);
count = N;

for i = 2:n
    
    patch_bits(1:N-l(i),:,i) = input_bits(count+1:count+N-l(i),:);
    count = count + N - l(i);
    parity{i} = double(rand(l(i),count)>0.5);
    patch_bits(N-l(i)+1:N,:,i) = mod(parity{i}*input_bits(1:count,:),2);
    
end 