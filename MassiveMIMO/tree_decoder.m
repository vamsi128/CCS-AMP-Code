% tree_decoder  Decodes a patched message encoded by tree_encoder.m
%
% output_bits     output_bits{i} is a B x k_i matrix of the k_i B-bit
%                 messages found in patch i
% l               length-n vector of number of parity check digits
%                 use the same as for tree_encoder
% parity          parity check codes: use the one outputted by tree-encoder
%
% found_bits       N x K matrix of the K repatched N-bit messages 
%
% Code is based on 'A Coupled Compressive Sensing Scheme for Unsourced 
% Multiple Access' by Amalladinne et al. 2018 (arXiv 1806.00138)

function found_bits = tree_decoder(output_bits,l,parity)

n = length(output_bits);
for i = 1:n
    k(i) = size(output_bits{i},2);
end
N = size(output_bits{1},1);

found_bits = [];
%keyboard
for i = 1:k(1)
    
    cand_bits = output_bits{1}(:,i);
 %   keyboard
    for j = 2:n
  %      keyboard
        cand_bits = [repmat(cand_bits,[1 k(j)]); kron(output_bits{j}(:,:),ones(1,size(cand_bits,2)))];
   %     keyboard
        parity_check = mod(parity{j}*cand_bits(1:end-l(j),:)+cand_bits(end-l(j)+1:end,:),2); 
        find_correct = sum(parity_check,1)<0.5;
        cand_bits = cand_bits(1:end-l(j),find_correct);
    %    keyboard
    end
    if size(cand_bits,2)==1
    found_bits = [found_bits cand_bits];
    end
        
end