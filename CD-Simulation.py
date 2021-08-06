import FactorGraphGeneration as FGG
import ccsinnercode as ccsic
import numpy as np

def estimate_bin_occupancy(Ka, dbid, rxK, K, s_n, GENIE_AIDED):
    """
    Estimate the number of users present in each bin. 
    :param Ka: total number of active users
    :param dbid: power allocated to bin occupancy estimation
    :param rxK: received vector indicating how many users are present in each bin
    :param K: true vector of # users/bin
    :param s_n: noise standard deviation
    :param GENIE_AIDED: boolean flag whether to return genie-aided estimates
    """

    # Set up MMSE Estimator
    pi = 1 / NUM_BINS                      # probability of selecting bin i (all bins are equally likely)
    Rzz = s_n**2 * np.eye(NUM_BINS)        # noise autocorrelation matrix
    Rbb = np.zeros((NUM_BINS, NUM_BINS))   # construct autocorrelation matrix for bin identification sequence
    for i in range(NUM_BINS):
        for j in range(NUM_BINS):
            Rbb[i, j] = Ka*pi*(1-pi) if i == j else -1*Ka*pi**2  # variance if diagonal entry, covariance if off diagonal
            Rbb[i, j] += (Ka * pi)**2      # add u_i*u_j to each entry to convert from covariance to autocorrelation
    print('Rbb: ')
    print(Rbb)
    
    # Construct MMSE matrix W
    Ryy = dbid**2 * Rbb + Rzz                           # autocorrelation of rx vector y (A matrix is identity)
    Ryb = dbid * Rbb                                    # cross-correlation of rx vector y and tx vector b 
    W_mmse = np.matmul(np.linalg.inv(Ryy), Ryb)         # LMMSE matrix W

    # LMMSE estimation
    Kht_mmse = np.matmul(W_mmse.conj().T, rxK)          # LMMSE estimate of binIdBits
    mmse = Kht_mmse*Ka/np.linalg.norm(Kht_mmse, ord=1)  # scale estimates to have correct L1 norm 
    Kht = np.maximum(1, np.ceil(mmse)).astype(int)      # take max of 1 and ceil of estimates
    print('True K: \t' + str(K))
    print('Estimated K: \t' + str(Kht))
    
    # Invoke Genie assistance if desired
    if GENIE_AIDED:
        Kht = K.copy()
        print('Genie-aided K: \t' + str(Kht))

    return Kht


# Simulation Parameters
Ka = 64                 # total number of users
NUM_BINS = 2            # number of bins employed
GENIE_AIDED = False     # flag of whether to produce genie-aided bin occupancy estimates
BlockDiagonal = False   # flag of whether to use block-diagonal sensing matrices

w = 128                 # payload size of each active user
N = 38400               # total number of channel uses (real d.o.f.)
L = 16                  # number of sections within code
M = 2**16               # length of each section
numAmpIter = 10         # number of AMP iterations
numBPIter = 1           # number of BP iterations to run
BPonOuterGraph = True   # flag of whether to run BP on outer graph within AMP denoiser
maxSims = 2             # number of trials to average over

EbNodB = 1.6            # Energy per bit in decibels
EbNo = 10**(EbNodB/10)  # Eb/No in linear scale
P = 2*w*EbNo/N
std = 1                 # noise standard deviation
errorRate = 0.0         # track error rate across simulations

# Assign power to occupancy estimation and data transmission tasks
pM = 80
dmsg = np.sqrt(N*P*N/(pM + N)/L) if NUM_BINS > 1 else np.sqrt(N*P/L)
dbid = np.sqrt(N*P*pM/(pM + N)) if NUM_BINS > 1 else 0
assert np.abs(L*dmsg**2 + dbid**2 - N*P) <= 1e-3, "Total power constraint violated."

# Average of maxSims trials
for idxsim in range(maxSims):
    print('Starting simulation %d of %d' % (idxsim + 1, maxSims))

    # Initialze outer codes
    OuterCodes = [FGG.Triadic8(16) for i in range(NUM_BINS)]

    # Generate messages for all Ka users
    usrmessages = np.random.randint(2, size=(Ka, w))

    # Split users into bins based on the first couple of bits in their messages
    w0 = int(np.ceil(np.log2(NUM_BINS)))
    binIds = np.matmul(usrmessages[:,0:w0], 2**np.arange(w0)[::-1]) if w0 > 0 else np.zeros(Ka)
    K = np.array([np.sum(binIds == i) for i in range(NUM_BINS)]).astype(int)

    # Group usrmessages by bin
    messages = [usrmessages[np.where(binIds == i)[0]] for i in range(NUM_BINS)]

    # Transmit bin identifier across channel
    rxK = dbid * K + np.random.randn(NUM_BINS) * std

    # Estimate the number of users per bin
    Kht = estimate_bin_occupancy(Ka, dbid, rxK, K, std, GENIE_AIDED)

    # Outer-encode user signals
    txcodewords = 0                                     # list of all codewords transmitted
    codewords = []                                      # list of codewords to transmitted by bin
    sTrue = []                                          # list of signals sent by various bins
    for i in range(NUM_BINS):
        # Outer encode each message
        cdwds = OuterCodes[i].encodemessages(messages[i]) if len(messages[i]) > 0 else [np.zeros(L*M)]
        for cdwd in cdwds:                              # ensure that each codeword is valid
            OuterCodes[i].testvalid(cdwd)
        codewords.append(cdwds)                         # add encoded messages to list of codewords
        txcodewords = np.vstack((txcodewords, cdwds)) if not np.isscalar(txcodewords) else cdwds.copy()

        # Combine codewords to form signal to transmit
        if K[i] == 0:                                   # do nothing if there are no users in this bin
            tmp = np.zeros(L*M).astype(np.float64)
        else:                                           # otherwise, add all codewords together
            tmp = np.sum(cdwds, axis=0)
        sTrue.append(tmp)                               # store true signal for future reference

    # Initialize inner code
    InnerCode = ccsic.CodedDemixingInnerCode(N, L/N*dmsg**2, std, Kht, NUM_BINS, BlockDiagonal,OuterCodes)

    # Inner encoding
    x = InnerCode.Encode(sTrue)

    # Transmit over channel
    y = x + np.random.randn(N, 1) * std

    # Inner decoding
    s = InnerCode.Decode(y, numAmpIter, BPonOuterGraph, numBPIter)

    # Prepare for outer graph decoding
    recoveredcodewords = dict()   
    matches = 0  

    # Graph-based outer decoder
    for idxbin in range(NUM_BINS):
        if Kht[idxbin] == 0: continue

        # Produce list of recovered codewords with their associated likelihoods
        recovered, likelihoods = OuterCodes[idxbin].decoder(s[idxbin], int(4*Kht[idxbin] + 5), includelikelihoods=True)

        # Compute what the first w0 bits should be based on bin number
        binIDBase2 = np.binary_repr(idxbin)
        binIDBase2 = binIDBase2 if len(binIDBase2) == w0 else (w0 - len(binIDBase2))*'0' + binIDBase2
        firstW0bits = np.array([binIDBase2[i] for i in range(len(binIDBase2))]).astype(int)

        # Enforce CRC consistency and add recovered codewords to data structure indexed by likelihood
        for idxcdwd in range(len(likelihoods)):
            
            # Extract first part of message from codeword
            firstinfosection = OuterCodes[idxbin].infolist[0] - 1
            sparsemsg = recovered[idxcdwd][firstinfosection*M:(firstinfosection+1)*M]

            # Find index of nonzero entry and convert to binary representation
            idxnonzero = np.where(sparsemsg > 0.0)[0][0]
            idxnonzerobin = np.binary_repr(idxnonzero)

            # Add trailing zeros to base-2 representation
            if len(idxnonzerobin) < 16:
                idxnonzerobin = (16 - len(idxnonzerobin))*'0' + idxnonzerobin

            # Extract first w0 bits and determine bin ID
            msgfirstW0bits = np.array([idxnonzerobin[i] for i in range(w0)]).astype(int)

            # Enforce CRC consistency
            if (msgfirstW0bits==firstW0bits).all() or (NUM_BINS == 1):
                recoveredcodewords[likelihoods[idxcdwd]] = np.hstack((recovered[idxcdwd], idxbin))
        
    # sort dictionary of recovered codewords in descending order of likelihood
    sortedcodewordestimates = sorted(recoveredcodewords.items(), key=lambda x: -x[0])

    # recover percentage of codewords based on SIC iteration
    sortedcodewordestimates = sortedcodewordestimates[0:Ka]

    # remove contribution of recovered users from received signal and prepare for PUPE computation
    codewordestimates = 0
    for idxcdwd in range(len(sortedcodewordestimates)):

        # add codeword to numpy array of rx codewords for PUPE computation
        codewordestimates = np.vstack((
                                codewordestimates, sortedcodewordestimates[idxcdwd][1][0:L*M]))  \
                                if not np.isscalar(codewordestimates) \
                                else sortedcodewordestimates[idxcdwd][1][0:L*M].copy()

    # Update number of matches
    matches += FGG.numbermatches(txcodewords, codewordestimates)
    print(str(matches) + ' matches')

# Compute error rate
errorRate += (Ka - matches) / (Ka * maxSims)
print('PUPE: ' + str(errorRate))