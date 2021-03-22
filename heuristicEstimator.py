import numpy as np
import FactorGraphGeneration as FGG
from pyfht import block_sub_fht

OuterCode1 = FGG.Triadic8(16)
OuterCode2 = FGG.Triadic8(16)

def sparc_codebook(L, M, n,P):
    Ax, Ay, _ = block_sub_fht(n, M, L, seed=None, ordering=None) # seed must be explicit
    def Ab(b):
        return Ax(b).reshape(-1, 1) / np.sqrt(n)
    def Az(z):
        return Ay(z).reshape(-1, 1) / np.sqrt(n) 
    return Ab, Az

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
    sHat = ( q*np.exp( -(r-d)**2 / (2*(tau**2)) ) \
            / ( q*np.exp( -(r-d)**2 / (2*(tau**2))) + (1-q)*np.exp( -r**2 / (2*(tau**2))) ) ).astype(float)
    return sHat

def dynamicDenoiser(r,p0,OuterCode,K,tau,d,numBPiter):
    """
    Args:
        r (float): Effective observation
        d (float): Signal amplitude
        tau (float): Standard deviation of noise
    """
    M = OuterCode.sparseseclength
    L = OuterCode.varcount

    # p0 = 1-(1-1/M)**K
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

    eAvg = 0.0

    for varnodeid in OuterCode.varlist:
        idx = varnodeid - 1
        # Beta[idx,:] = OuterCode.getestimate(varnodeid)
        Beta[idx,:] = OuterCode.getextrinsicestimate(varnodeid)
        mu[idx*M:(idx+1)*M] = approximateVector(Beta[idx,:], K).reshape(-1,1)

    return mu

def amp_effective_observation(z, s, P, L, Az):
    
    # required parameters
    n = z.size
    d = np.sqrt(n*P/L)

    # Compute effective observation
    r = (d*s + Az(z.flatten()))

    return r

def amp_estimate_k_distribution(s1, s2, L, m, Ka, idxit):
    # m = s1.size

    r1 = 0.0
    r2 = 0.0

    for i in range(L):
        w1 = s1[i*m:(i+1)*m]
        w2 = s2[i*m:(i+1)*m]

        idxw1 = np.argpartition(w1, -Ka, axis=0)[-Ka:]
        idxw2 = np.argpartition(w2, -Ka, axis=0)[-Ka:]

        sw1 = np.sum(w1[idxw1])
        sw2 = np.sum(w2[idxw2])
        
        r1t = sw1/(sw1+sw2)
        r2t = 1 - r1t

        r1 += r1t/L
        r2 += r2t/L

    # a = 2**2

    kr1 = Ka*r1
    kr2 = Ka*r2

    k1ht = round(kr1)
    k2ht = round(kr2)

    print('K1 estimate: ', k1ht)
    print('K2 estimate: ', k2ht)

    # compute confidence
    kr1d = kr1 % 1
    kr2d = kr2 % 1
    cl = np.abs(kr1d-kr2d)
    confidence = (cl >= 0.3)
    print('Confidence level: ', cl)
    print('Confident? ', confidence)

    return (k1ht, k2ht, confidence)

def amp_state_update(r, z, P, L, K, Ka, numBPiter, OuterCode, confident):

    """
    Args:
        s: State update through AMP composite iteration
        z: Residual update through AMP composite iteration
        tau (float): Standard deviation of noise
        mu: Product of messages from adjoining factors
    """
    n = z.size
    m = r.size
    d = np.sqrt(n*P/L)

    # Compute tau online using the residual
    tau = np.sqrt(np.sum(z**2)/n)

    # Uninformative prior
    if not confident:
        p0 = 1-(1-1/(2*m))**Ka 
    else:
        p0 = 1-(1-1/m)**K

    # Compute updated state
    mu = dynamicDenoiser(r, p0, OuterCode, K, tau, d, numBPiter)
    s = pme0(mu, r, d, tau)
        
    return s

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

def simulateSingleClass(EbNodB):
    simCount = 100 # number of simulations
    msgDetected1=0
    msgDetected2=0
    K1Sum = 0
    K2Sum = 0
    numMC = 0

    for simIndex in range(simCount):

        Ka = 64
        K1 = 0

        K1 = np.sum(np.random.randn(Ka) > 0)
        K2 = Ka - K1

        K1Sum += K1
        K2Sum += K2

        print('K1: ', K1)
        print('K2: ', K2)

        B1=128 # Payload size of every active user in group 1
        B2=128 # Payload size of every active user in group 2

        L1=16 # Number of sections/sub-blocks in group 1
        L2=16 # Number of sections/sub-blocks in group 2

        n=38400 # Total number of channel uses (real d.o.f)
        T=10 # Number of AMP iterations
        J1=16  # Length of each coded sub-block
        J2=16  # Length of each coded sub-block
        M1=2**J1 # Length of each section
        M2=2**J2 # Length of each section

        numBPiter = 1 # Number of BP iterations on outer code. 1 seems to be good enough & AMP theory including state evolution valid only for one BP iteration
        # EbNodB = 2.4 # Energy per bit. With iterative extension, operating EbN0 falls to 2.05 dB for 25 users with 1 round SIC
        

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

        K1ht = Ka
        K2ht = Ka
        confident = False

        for t in range(T):
            # compute effective observations
            r1 = amp_effective_observation(z ,s1, P1, L1, Az1)
            r2 = amp_effective_observation(z, s2, P2, L2, Az2)

            # AMP state updates
            s1 = amp_state_update(r1, z, P1, L1, K1ht, Ka, numBPiter, OuterCode1, confident)
            s2 = amp_state_update(r2, z, P2, L2, K2ht, Ka, numBPiter, OuterCode2, confident)
            
            # Estimate K1, K2
            if not confident:
                k1ht, k2ht, confident = amp_estimate_k_distribution(s1, s2, L1, M1, Ka, t)
                if confident:  # employ estimate after 2 amp iterations
                    K1ht = k1ht
                    K2ht = k2ht
                    if k1ht != K1:
                        numMC += 1
                        print('ERROR: K1, K2 misestimated.')

            # AMP residual
            z = amp_residual(y, z, s1, s2, d1, d2, Ab1, Ab2)

        # continue

        # raise Exception()
        print('Graph Decode')
        
        # Decoding with Graph
        originallist1 = codewords1.copy()
        originallist2 = codewords2.copy()
        qq1 = int(K1ht * 1.5) if confident else int(Ka*1.5 / 2)
        qq2 = int(K2ht * 1.5) if confident else int(Ka*1.5 / 2)
        recoveredcodewords1 = OuterCode1.decoder(s1, qq1)
        recoveredcodewords2 = OuterCode2.decoder(s2, qq2)

        # Calculation of per-user prob err
        matches1 = FGG.numbermatches(originallist1, recoveredcodewords1)
        matches2 = FGG.numbermatches(originallist2, recoveredcodewords2)
        
        print('Group 1: ' + str(matches1) + ' out of ' + str(K1))
        print('Group 2: ' + str(matches2) + ' out of ' + str(K2))
        msgDetected1 = msgDetected1 + matches1
        msgDetected2 = msgDetected2 + matches2
        
    # errorRate1= (K1*simCount - msgDetected1)/(K1*simCount)
    # errorRate2= (K2*simCount - msgDetected2)/(K2*simCount)
    errorRate1 = (K1Sum - msgDetected1) / K1Sum
    errorRate2 = (K2Sum - msgDetected2) / K2Sum

    print("Per user probability of error (Group 1) = ", errorRate1)
    print("Per user probability of error (Group 2) = ", errorRate2)
    print("Average PUPE =  ", 0.5*(errorRate1 + errorRate2))

    return errorRate1, errorRate2, 0.5*(errorRate1 + errorRate2), numMC


EbNos = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
e1 = []
e2 = []
eavg = []
emc = []

for snr in EbNos:
    a, b, c, d = simulateSingleClass(snr)
    e1.append(a)
    e2.append(b)
    eavg.append(c)
    emc.append(d)

print(e1)
print(e2)
print(eavg)
print(emc)

# np.savetxt('e1.txt', e1)
# np.savetxt('e2.txt', e2)
np.savetxt('2_class_heuristic_estimator_eavg.txt', eavg)
np.savetxt('2_clas__mc.txt', emc)
