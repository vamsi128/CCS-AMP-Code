import numpy as np
import FactorGraphGeneration as FGG
from pyfht import block_sub_fht

# Initialize Graphs
OuterCode1 = FGG.Triadic8(15)
OuterCode2 = FGG.Disco3(15)

# Create SPARC Codebook
def sparc_codebook(L, M, n,P):
    Ax, Ay, _ = block_sub_fht(n, M, L, seed=None, ordering=None) # seed must be explicit
    def Ab(b):
        return Ax(b).reshape(-1, 1) / np.sqrt(n)
    def Az(z):
        return Ay(z).reshape(-1, 1) / np.sqrt(n) 
    return Ab, Az

# Vector approximation
def approximateVector(x, K):    

    # normalize initial value of x
    xOrig = x / np.linalg.norm(x, ord=1)
    
    # create vector to hold best approximation of x
    xHt = xOrig.copy()
    u = np.zeros(len(xHt))
    
    # run approximation algorithm
    while np.amax(xHt) > (1/K):
        minIndices = np.argmin([(1/K)*np.ones(xHt.shape), xHt], axis=0)
        xHt = np.min([(1/K)*np.ones(xHt.shape), xHt], axis=0)
        
        deficit = 1 - np.linalg.norm(xHt, ord=1)
        
        if deficit > 0:
            mIxHtNorm = np.linalg.norm((xHt*minIndices), ord=1)
            scaleFactor = (deficit + mIxHtNorm) / mIxHtNorm
            xHt = scaleFactor*(minIndices*xHt) + (1/K)*(np.ones(xHt.shape) - minIndices)

    # return admissible approximation of x
    return xHt

# Posterior Mean Estimator
def pme0(q, r, d, tau):
    """Posterior mean estimator (PME)
    
    Args:
        q (float): Prior probability
        r (float): Effective observation
        d (float): Signal amplitude
        tau (float): Standard deviation of noise
    Returns:
        sHat (float): Probability s is one
    
    """
    sHat = ( q*np.exp( -(r-d)**2 / (2*(tau**2)) )             / ( q*np.exp( -(r-d)**2 / (2*(tau**2))) + (1-q)*np.exp( -r**2 / (2*(tau**2))) ) ).astype(float)
    return sHat

# Dynamic Denoiser
def dynamicDenoiser(r,OuterCode,K,tau,d,numBPiter):
    """
    Args:
        r (float): Effective observation
        d (float): Signal amplitude
        tau (float): Standard deviation of noise
    """
    M = OuterCode.sparseseclength
    L = OuterCode.varcount

    p0 = 1-(1-1/M)**K
    p1 = p0*np.ones(r.shape, dtype=float)
    mu = np.zeros(r.shape, dtype=float)

    # Compute local estimate (lambda) based on effective observation using PME.
    localEstimates = pme0(p0, r, d, tau)
    
    # Reshape local estimate (lambda) into an LxM matrix
    Beta = localEstimates.reshape(L,-1)
    OuterCode.reset()
    for varnodeid in OuterCode.varlist:
        idx = varnodeid - 1
        Beta[idx,:] = approximateVector(Beta[idx,:], K)
        OuterCode.setobservation(varnodeid, Beta[idx,:])
    
    for iteration in range(numBPiter):
        OuterCode.updatechecks()
        OuterCode.updatevars()

    for varnodeid in OuterCode.varlist:
        idx = varnodeid - 1
        # Beta[idx,:] = OuterCode.getestimate(varnodeid)
        Beta[idx,:] = OuterCode.getextrinsicestimate(varnodeid)
        mu[idx*M:(idx+1)*M] = approximateVector(Beta[idx,:], K).reshape(-1,1)

    return mu

# AMP State Update
def amp_state_update(z, s, P, L, Ab, Az, K, numBPiter, OuterCode):

    """
    Args:
        s: State update through AMP composite iteration
        z: Residual update through AMP composite iteration
        tau (float): Standard deviation of noise
        mu: Product of messages from adjoining factors
    """
    n = z.size
    d = np.sqrt(n*P/L)

    # Compute tau online using the residual
    tau = np.sqrt(np.sum(z**2)/n)

    # Compute effective observation
    r = (d*s + Az(z.flatten()))

    # Compute updated state
    mu = dynamicDenoiser(r, OuterCode, K, tau, d, numBPiter)
    s = pme0(mu, r, d, tau)
        
    return s

# AMP Residual
def amp_residual(y, z, s1, s2, d1, d2, Ab1, Ab2):
    """
    Args:
        s1: State update through AMP composite iteration
        s2: State update through AMP composite iteration
        y: Original observation
        tau (float): Standard deviation of noise
    """
    n = y.size
    
    # Compute tau online using the residual
    tau = np.sqrt(np.sum(z**2)/n)

    # Compute residual
    Onsager1 = (d1**2)*(np.sum(s1) - np.sum(s1**2))
    Onsager2 = (d2**2)*(np.sum(s2) - np.sum(s2**2))   
    z_plus = y - d1*Ab1(s1) - d2*Ab2(s2)+ (z/(n*tau**2))*(Onsager1 + Onsager2)
    
    return z_plus


# Simulation
def simulate(EbNodB):
    K1 = 25 # Number of active users in group 1
    K2 = 25 # Number of active users in group 2

    B1=120 # Payload size of every active user in group 1
    B2=165 # Payload size of every active user in group 2

    L1=OuterCode1.varcount # Number of sections/sub-blocks in group 1
    L2=OuterCode2.varcount # Number of sections/sub-blocks in group 2

    n=38400 # Total number of channel uses (real d.o.f)
    T=10 # Number of AMP iterations
    J1=OuterCode1.seclength  # Length of each coded sub-block
    J2=OuterCode2.seclength  # Length of each coded sub-block
    M1=OuterCode1.sparseseclength # Length of each section
    M2=OuterCode2.sparseseclength # Length of each section

    numBPiter = 1 # Number of BP iterations on outer code. 1 seems to be good enough & AMP theory including state evolution valid only for one BP iteration
    # EbNodB = 3 # Energy per bit. With iterative extension, operating EbN0 falls to 2.05 dB for 25 users with 1 round SIC
    simCount = 100 # number of simulations

    # EbN0 in linear scale
    EbNo = 10**(EbNodB/10)
    P1 = 2*B1*EbNo/n
    P2 = 2*B2*EbNo/n
    s_n = 1

    # We assume equal power allocation for all the sections. Code has to be modified a little to accomodate non-uniform power allocations
    Phat1 = n*P1/L1
    Phat2 = n*P2/L2
    d1 = np.sqrt(n*P1/L1)
    d2 = np.sqrt(n*P2/L2)

    # Vectors of PUPE results for confidence intervals
    group1PUPE = np.zeros(simCount)
    group2PUPE = np.zeros(simCount)

    # Reset message detected variables
    msgDetected1=0
    msgDetected2=0

    for simIndex in range(simCount):
        print('Simulation Number: ' + str(simIndex))
        
        # Generate active users message sequences
        messages1 = np.random.randint(2, size=(K1, B1))
        messages2 = np.random.randint(2, size=(K2, B2))

        # Outer-encode the message sequences
        codewords1 = OuterCode1.encodemessages(messages1)
        for codeword1 in codewords1:
            OuterCode1.testvalid(codeword1)
        codewords2 = OuterCode2.encodemessages(messages2)
        for codeword2 in codewords2:
            OuterCode2.testvalid(codeword2)

        # Convert indices to sparse representation
        # sTrue: True state
        sTrue1 = np.sum(codewords1, axis=0)
        sTrue2 = np.sum(codewords2, axis=0)

        # Generate the binned SPARC codebook
        Ab1, Az1 = sparc_codebook(L1, M1, n, P1)
        Ab2, Az2 = sparc_codebook(L2, M2, n, P2)
        
        # Generate our transmitted signal X
        x = d1*Ab1(sTrue1) + d2*Ab2(sTrue2)
        
        # Generate random channel noise and thus also received signal y
        noise = np.random.randn(n, 1) * s_n
        y = (x + noise)

        z = y.copy()
        s1 = np.zeros((L1*M1, 1))
        s2 = np.zeros((L2*M2, 1))

        for t in range(T):
            s1 = amp_state_update(z, s1, P1, L1, Ab1, Az1, K1, numBPiter, OuterCode1)
            s2 = amp_state_update(z, s2, P2, L2, Ab2, Az2, K2, numBPiter, OuterCode2)
            z = amp_residual(y, z, s1, s2, d1, d2, Ab1, Ab2)

        print('Graph Decode')
        
        # Decoding with Graph
        originallist1 = codewords1.copy()
        originallist2 = codewords2.copy()
        recoveredcodewords1 = OuterCode1.decoder(s1, 2*K1)
        recoveredcodewords2 = OuterCode2.decoder(s2, 2*K2)

        # Calculation of per-user prob err
        simMsgDetected1 = 0
        simMsgDetected2 = 0
        matches1 = FGG.numbermatches(originallist1, recoveredcodewords1)
        matches2 = FGG.numbermatches(originallist2, recoveredcodewords2)
        
        print('Group 1: ' + str(matches1) + ' out of ' + str(K1))
        print('Group 2: ' + str(matches2) + ' out of ' + str(K2))
        msgDetected1 = msgDetected1 + matches1
        msgDetected2 = msgDetected2 + matches2

        group1PUPE[simIndex] = (K1-matches1) / K1
        group2PUPE[simIndex] = (K2-matches2) / K2
        
    errorRate1= (K1*simCount - msgDetected1)/(K1*simCount)
    errorRate2= (K2*simCount - msgDetected2)/(K2*simCount)

    print("Per user probability of error (Group 1) = ", errorRate1)
    print("Per user probability of error (Group 2) = ", errorRate2)

    # Print results to text file
    np.savetxt('group1_snr_'+str(int(EbNodB*10))+'_pupe.txt', group1PUPE)
    np.savetxt('group2_snr_'+str(int(EbNodB*10))+'_pupe.txt', group2PUPE)

    return errorRate1, errorRate2


# Run Simulation
EbNodBs = [2.8, 3.0, 3.2]
peg1 = np.zeros(len(EbNodBs))
peg2 = np.zeros(len(EbNodBs))

for idxsnr in range(len(EbNodBs)):
    g1, g2 = simulate(EbNodBs[idxsnr])
    peg1[idxsnr] = g1
    peg2[idxsnr] = g2

np.savetxt('210308_avg_pupe_g1.txt', peg1)
np.savetxt('210305_avg_pupe_g2.txt', peg2)