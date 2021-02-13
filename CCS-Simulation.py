import ccsfg
import FactorGraphGeneration as FG
import ccsinnercode as ccsic
import numpy as np

# Simulation Parameters
Ka = 25                 # Number of active users
w = 128                 # Payload size of each active user (per user message length)
N = 38400               # Total number of channel uses (real d.o.f)
L = 16                  # Number of sections/sub-blocks
vl = 16                 # Length of each coded sub-block
ml = 2**vl              # Length of each section of m
listSize = Ka+10        # List size retained for each section after AMP converges

numAmpIter = 6          # Number of AMP iterations
numBPIter = 1           # Number of BP iterations to perform
BPonOuterGraph = True  # Indicates whether to perform BP on outer code.  If 0, AMP uses Giuseppe's uninformative prior
EbNodB = 2.4            # Energy per bit in decibels
maxSims = 2            # Number of Simulations to Run
EbNo = 10**(EbNodB/10)  # Eb/No in linear scale
P = 2*w*EbNo/N          # transmit power
std = 1                 # Noise standard deviation
Phat = N*P/L            # Power estimate
errorRate = 0.0         # track error rate across simulations

# Initialize CCS-AMP Graph
Graph = FG.Graph8()

# Run CCS-AMP maxSims times
for idxsim in range(maxSims):
    print('Starting simulation %d of %d' % (idxsim + 1, maxSims))
    
    # Reset the graph
    Graph.reset()
    
    # Set up Inner Encoder/Decoder
    InnerCode = ccsic.BlockDiagonalInnerCode(L, ml, N, P, std, Ka)
    
    # Generate random messages for Ka active users
    txBits = np.random.randint(low=2, size=(Ka, w))
    
    # Outer LDPC Encoder
    txMsg = Graph.encodemessages(txBits)
    x = Graph.encodesignal(txBits)
    
    # Inner CS Encoder
    x = InnerCode.Encode(Phat, x)
    
    # Transmit x over channel
    y = (x + (np.random.randn(N, 1) * std)).reshape(-1, 1)

    # Inner CS Decoder
    xHt, tau_evolution = InnerCode.Decode(y, numAmpIter, BPonOuterGraph, numBPIter, Graph)

    # Outer LDPC Decoder (Message Disambiguation)
    txMsgHt = FG.decoder(Graph, xHt, listSize)
    
    # Calculate PUPE
    errorRate += (Ka - FG.numbermatches(txMsg, txMsgHt)) / (Ka * maxSims)

# Display Simulation Results
print("Per user probability of error = %3.6f" % errorRate)