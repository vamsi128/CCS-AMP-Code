function G = createG(systemParameters)

L = systemParameters.L;
parityLengths = systemParameters.parityLengths;
messageLengths = systemParameters.messageLengths;
G = cell(L,1);

for l=2:L % 1st section has no parity bits
    % Generate the Rademacher matrix corresponding to section l
    G{l} = randi([0,1],(sum(messageLengths(1:l-1))),parityLengths(l));
end

end
