function survivingPaths=tree_decode_update_suriving_paths(survivingPaths,path,i,j,Trans_cell_estimate,SystemParameters,parity)
A=survivingPaths{j,i};


for k=1:size(Trans_cell_estimate{i},1)
    index = parity_check(path,k,Trans_cell_estimate,parity,SystemParameters);
    if(index==1)
        new = [path k];
        A=[A;new];
    end
    
end



survivingPaths{j,i}=A;

end