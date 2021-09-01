import numpy as np
from pyfht import block_sub_fht
import FactorGraphGeneration as FGG

class CodedDemixingSimulation:
    """
    Class @CodedDemixingSimulation runs a coded demixing simulation with a variety
    of parameters including tree code vs LDPC code, AMP vs ADMM, and block-diagonal 
    vs dense sensing matrix.

    Sample use of Class @CodedDemixingSimulation:
    >>> myCDSim = CodedDemixing.CodedDemixingSimulation(...)
    >>> myCDSim.setParameters(...)
    >>> errorRate = myCDSim.simulate(...)
    """

    # Methods intended to be called externally
    def __init__(self, OuterCodeType, InnerEncoderType, InnerDecoderType, DenoiserType, SensingMatrixType):
        """
        Initialize node of type @class CodedDemixingSimulation
        :param OuterCodeType: specify between 'TreeCode' and 'LDPCCode'
        :param InnerEncoderType: specify between 'FHT' and other options
        :param InnerDecoderType: specify between 'AMP' and 'ADMM'
        :param DenoiserType: specify between 'PME' and other options
        :param SensingMatrixType: specify between 'Dense' and 'BlockDiagonal'
        """

        self.__OuterCodeType = OuterCodeType
        self.__InnerEncoderType = InnerEncoderType
        self.__InnerDecoderType = InnerDecoderType
        self.__DenoiserType = DenoiserType
        self.__SensingMatrixType = SensingMatrixType

    def setParameters(self, w, n, L, M, numAMPIter, numBPIter, BPonOuterGraph, 
                      pM, numBins, genieAided):
        """
        Set simulation parameters.
        :param w: payload size of each user
        :param n: number of channel uses
        :param L: number of sections
        :param M: length of each section
        :param numAMPIter: number of AMP iterations
        :param numBPIter: number of BP iterations
        :param BPonOuterGraph: boolean flag of whether to perform BP on outer factor graph
        :param pM: power multiplier for occupancy estimation 
        :param numBins: number of coded demixing bins
        :param genieAided: boolean flag of whether to use genie aided bin estimates
        """

        # Set appropriate parameters
        self.__w = w
        self.__n = n
        self.__L = L
        self.__M = M
        self.__numAMPIter = numAMPIter
        self.__numBPIter = numBPIter
        self.__BPonOuterGraph = BPonOuterGraph
        self.__pM = pM
        self.__numBins = numBins
        self.__genieAided = genieAided

    def simulate(self, Ktot, EbNodB, numSims):
        """
        Run a coded demixing simulation.
        :param Ktot: total number of active users
        :param EbNodB: Eb/No in dB scale
        :param numSims: number of trials to average the results over
        """

        # Create alias for certain parameters
        w = self.__w
        n = self.__n
        L = self.__L
        pM = self.__pM
        numBins = self.__numBins

        # Compute required power parameters
        EbNo = 10**(EbNodB/10)
        P = 2*w*EbNo/n
        std = 1
        dmsg = np.sqrt(n*P*n/(pM + n)/L) if numBins > 1 else np.sqrt(n*P/L)
        dbid = np.sqrt(n*P*pM/(pM + n)) if numBins > 1 else 0
        assert np.abs(L*dmsg**2 + dbid**2 - n*P) <= 1e-3, "Total power constraint violated."

        # Compute empirical PUPE averaged over numSims trials
        errorRate = 0.0
        matches = 0
        for idxsim in range(numSims):
            print('Starting simulation %d of %d' % (idxsim + 1, numSims))

            # Generate messages for all Ktot users
            usrmessages = np.random.randint(2, size=(Ktot, w))

            # Split users into bins based on the first couple of bits in their messages
            w0 = int(np.ceil(np.log2(numBins)))
            self.__w0 = w0
            binIds = np.matmul(usrmessages[:,0:w0], 2**np.arange(w0)[::-1]) if w0 > 0 else np.zeros(K)
            K = np.array([np.sum(binIds == i) for i in range(numBins)]).astype(int)
            self.__K = K.copy()

            # Group usrmessages by bin
            messages = [usrmessages[np.where(binIds == i)[0]] for i in range(numBins)]

            # Transmit bin identifier across channel
            rxK = dbid * K + np.random.randn(numBins) * std

            # Estimate the number of users per bin
            Kht = self.estimateBinOccupancy(Ktot, dbid, rxK, std)
            self.__Kht = Kht.copy()

            # Outer-encode user signals
            txcodewords, sTrue = self.outerEncode(messages)

            # Inner-encode user signals
            x = self.innerEncode(dmsg, sTrue)

            # Transmit over channel
            y = x + std*np.random.randn(n, 1)

            # Inner-decoding
            s = self.innerDecode(y, dmsg)

            # Outer-decoding
            codewordestimates = self.outerDecode(s, Ktot)

            # Compute number of matches
            matches = FGG.numbermatches(txcodewords, codewordestimates)
            errorRate += (Ktot - matches) / (Ktot * numSims)
            print(str(matches) + ' matches')

        return errorRate

    
    # Outer Encoding Methods
    def outerEncode(self, messages):
        """
        Perform outer encoding of codewords
        :param messages: messages to be outer-encoded
        """
        
        # Outer-encoding
        if self.__OuterCodeType=='LDPCCode':
            return self.ldpcOuterEncode(messages)
        elif self.__OuterCodeType=='TreeCode':
            return self.treeOuterEncode(messages)

    def ldpcOuterEncode(self, messages):
        """
        Perform outer encoding of codewords
        :param messages: messages to be outer-encoded
        """

        # Create aliases for several parameters
        numBins = self.__numBins
        L = self.__L
        M = self.__M
        K = self.__K

        # Establish important parameters
        self.__OuterCodes = [FGG.Triadic8(16) for i in range(numBins)]
        txcodewords = 0                                     # list of all codewords transmitted
        codewords = []                                      # list of codewords to transmitted by bin
        sTrue = []                                          # list of signals sent by various bins

        # Outer encode each message
        for i in range(numBins):
            cdwds = self.__OuterCodes[i].encodemessages(messages[i]) if len(messages[i]) > 0 else [np.zeros(L*M)]
            for cdwd in cdwds:                              # ensure that each codeword is valid
                self.__OuterCodes[i].testvalid(cdwd)
            codewords.append(cdwds)                         # add encoded messages to list of codewords
            txcodewords = np.vstack((txcodewords, cdwds)) if not np.isscalar(txcodewords) else cdwds.copy()

            # Combine codewords to form signal to transmit
            if K[i] == 0:                                   # do nothing if there are no users in this bin
                tmp = np.zeros(L*M).astype(np.float64)
            else:                                           # otherwise, add all codewords together
                tmp = np.sum(cdwds, axis=0)
            sTrue.append(tmp)                               # store true signal for future reference

        
        return txcodewords, sTrue

    def treeOuterEncode(self, messages):
        # FIXME: this method needs to be built
        return -1


    # Outer Decoding Methods
    def outerDecode(self, s, Ktot):
        """
        Perform outer-decoding
        """

        if self.__OuterCodeType=='LDPCCode':
            return self.ldpcOuterDecode(s, Ktot)
        elif self.__OuterCodeType == 'TreeCode':
            return self.treeOuterDecode(s, Ktot)

    def ldpcOuterDecode(self, s, Ktot):
        """
        Graph-based LDPC code outer decoder
        :param s: array to decode users from
        :param Ktot: total number of active users
        """

        # create aliases for certain parameters
        numBins = self.__numBins
        M = self.__M
        L = self.__L
        w0 = self.__w0
        Kht = self.__Kht

        # Prepare for outer graph decoding
        recoveredcodewords = dict()   
        matches = 0  

        # Graph-based outer decoder
        for idxbin in range(numBins):
            if Kht[idxbin] == 0: continue

            # Produce list of recovered codewords with their associated likelihoods
            recovered, likelihoods = self.__OuterCodes[idxbin].decoder(s[idxbin], int(4*Kht[idxbin] + 5), includelikelihoods=True)

            # Compute what the first w0 bits should be based on bin number
            binIDBase2 = np.binary_repr(idxbin)
            binIDBase2 = binIDBase2 if len(binIDBase2) == w0 else (w0 - len(binIDBase2))*'0' + binIDBase2
            firstW0bits = np.array([binIDBase2[i] for i in range(len(binIDBase2))]).astype(int)

            # Enforce CRC consistency and add recovered codewords to data structure indexed by likelihood
            for idxcdwd in range(len(likelihoods)):
                
                # Extract first part of message from codeword
                firstinfosection = self.__OuterCodes[idxbin].infolist[0] - 1
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
                if (msgfirstW0bits==firstW0bits).all() or (numBins == 1):
                    recoveredcodewords[likelihoods[idxcdwd]] = np.hstack((recovered[idxcdwd], idxbin))
            
        # sort dictionary of recovered codewords in descending order of likelihood
        sortedcodewordestimates = sorted(recoveredcodewords.items(), key=lambda x: -x[0])

        # recover percentage of codewords based on SIC iteration
        sortedcodewordestimates = sortedcodewordestimates[0:Ktot]

        # remove contribution of recovered users from received signal and prepare for PUPE computation
        codewordestimates = 0
        for idxcdwd in range(len(sortedcodewordestimates)):

            # add codeword to numpy array of rx codewords for PUPE computation
            codewordestimates = np.vstack((
                                    codewordestimates, sortedcodewordestimates[idxcdwd][1][0:L*M]))  \
                                    if not np.isscalar(codewordestimates) \
                                    else sortedcodewordestimates[idxcdwd][1][0:L*M].copy()

        return codewordestimates

    def treeOuterDecode(self, s, Ktot):
        # FIXME: this method needs to be built
        return -1


    # Inner Encoding Methods
    def innerEncode(self, dmsg, sTrue):
        """
        Perform inner (compressed-sensing) encoding of codewords
        :param dmsg: accounts for transmit power
        :param sTrue: codewords to be inner-encoded
        """

        if self.__InnerEncoderType == 'FHT':
            return self.fhtInnerEncode(dmsg, sTrue)
        # else:
        #     return self.otherInnerEncode(dmsg, sTrue)

    def fhtInnerEncode(self, dmsg, sTrue):
        """
        FHT-based inner encoding of signal
        :param dmsg: amplitude of signal
        :param sTrue: codewords to inner encode
        """

        # Initialize important parameters
        x = np.zeros(self.__n).reshape((-1, 1))
        self.__Ab = []
        self.__Az = []

        # Create sensing matrices for each bin
        for idxbin in range(self.__numBins):
            tmpn = self.__n if self.__SensingMatrixType == 'Dense' else int(self.__n/self.__L)
            a, b = self.sparcCodebook(tmpn)
            self.__Ab.append(a)
            self.__Az.append(b)

        # Inner-encode each codeword 
        for idxbin in range(self.__numBins):

            if self.__SensingMatrixType == 'Dense':
                x += dmsg * self.__Ab[idxbin](sTrue[idxbin])
            
            elif self.__SensingMatrixType == 'BlockDiagonal':
                M, nbr = self.__M, self.__numBlockRows
                for idxblock in range(self.__L):
                    x[idxblock*nbr:(idxblock+1)*nbr] += dmsg * self.__Ab[idxbin](sTrue[idxbin][idxblock*M:(idxblock+1)*M])
        
        # return result
        return x


    # Inner Decoding Methods
    def innerDecode(self, y, dmsg):
        """
        Perform compressed-sensing recovery of signals
        :param y: received signal
        :param dmsg: signal amplitude
        """

        # perform inner decoding
        if self.__InnerDecoderType=='AMP':
            return self.ampInnerDecode(y, dmsg)
    
    def ampInnerDecode(self, y, dmsg):
        """
        AMP-based inner decoding
        :param y: received signal
        :param dmsg: signal amplitude
        """

        # Initialize important parameters
        L = self.__L
        M = self.__M
        numBins = self.__numBins

        # Prepare for inner AMP decoding
        z = y.copy()
        s = [np.zeros((L*M, 1)) for i in range(numBins)]

        # AMP Inner decoding
        for idxampiter in range(self.__numAMPIter):

            # Update the state of each bin individually
            s = [self.ampStateUpdate(z, s[i], dmsg, self.__Az[i], self.__Kht[i], self.__OuterCodes[i]) for i in range(numBins)]

            # compute residual jointly
            z = self.jointAMPResidual(y, z, s, dmsg)

        # return result
        return s


    # Helper Methods
    def estimateBinOccupancy(self, Ka, dbid, rxK, s_n):
        """
        Estimate the number of users present in each bin. 
        :param Ka: total number of active users
        :param dbid: power allocated to bin occupancy estimation
        :param rxK: received vector indicating how many users are present in each bin
        :param s_n: noise standard deviation
        :param GENIE_AIDED: boolean flag whether to return genie-aided estimates
        """

        # Create aliases for various parameters
        numBins = self.__numBins
        genieAided = self.__genieAided
        K = self.__K

        # Set up MMSE Estimator
        pi = 1 / numBins                      # probability of selecting bin i (all bins are equally likely)
        Rzz = s_n**2 * np.eye(numBins)        # noise autocorrelation matrix
        Rbb = np.zeros((numBins, numBins))   # construct autocorrelation matrix for bin identification sequence
        for i in range(numBins):
            for j in range(numBins):
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
        if genieAided:
            Kht = K.copy()
            print('Genie-aided K: \t' + str(Kht))

        return Kht
    
    def sparcCodebook(self, n):
        """
        Create sensing matrix via randomly sampling the rows of Hadamard matrices
        :param n: number of rows in matrix
        """

        # Initialize important parameters
        M = self.__M
        L = self.__L if self.__SensingMatrixType == 'Dense' else 1
        self.__numBlockRows = n

        # Create SPARC-like codebook
        Ax, Ay, _ = block_sub_fht(n, M, L, seed=None, ordering=None) # seed must be explicit
        def Ab(b):
            return Ax(b).reshape(-1, 1) / np.sqrt(n)
        def Az(z):
            return Ay(z).reshape(-1, 1) / np.sqrt(n) 
        return Ab, Az

    def approximateVector(self, x, K):   
        """
        Approximate factor graph message by enforcing certain constraints. 
        :param x: vector to approximate
        :param K: number of users in bin
        """ 

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

    def pme(self, q, r, d, tau):
        """
        Posterior mean estimator (PME)
        :param q: prior probability
        :param r: effective observation
        :param d: signal amplitude
        :param tau: noise standard deviation
        """
        sHat = ( q*np.exp( -(r-d)**2 / (2*(tau**2)) ) \
                / ( q*np.exp( -(r-d)**2 / (2*(tau**2))) + (1-q)*np.exp( -r**2 / (2*(tau**2))) ) ).astype(float)
        return sHat

    def computePrior(self, r, OuterCode, K, tau, dmsg):
        """
        Runs BP on outer factor graph to obtain informative priors
        :param r: AMP residual
        :param OuterCode: factor graph used in outercode
        :param K: number of users in a bin
        :param tau: noise standard deviation
        :param dmsg: signal amplitude
        """

        # Compute relevant parameters
        M = OuterCode.sparseseclength
        L = OuterCode.varcount
        numBPiter = self.__numBPIter
        p0 = 1-(1-1/M)**K
        p1 = p0*np.ones(r.shape, dtype=float)
        mu = np.zeros(r.shape, dtype=float)

        # Compute local estimate (lambda) based on effective observation using PME.
        localEstimates = np.zeros(r.shape)
        for idxblock in range(L):
            localEstimates[idxblock*M:(idxblock+1)*M] = self.denoiser(p0, r[idxblock*M:(idxblock+1)*M], dmsg, tau[idxblock])
        
        # Reshape local estimate (lambda) into an LxM matrix
        Beta = localEstimates.reshape(L,-1)

        # Initialize outer factor graph
        OuterCode.reset()
        for varnodeid in OuterCode.varlist:
            idx = varnodeid - 1
            Beta[idx,:] = self.approximateVector(Beta[idx,:], K)
            OuterCode.setobservation(varnodeid, Beta[idx,:])
        
        # Run BP on outer factor graph
        for iteration in range(numBPiter):
            OuterCode.updatechecks()
            OuterCode.updatevars()

        # Extract and return pertinent information from outer factor graph
        for varnodeid in OuterCode.varlist:
            idx = varnodeid - 1
            Beta[idx,:] = OuterCode.getextrinsicestimate(varnodeid)
            mu[idx*M:(idx+1)*M] = 1 - (1 - self.approximateVector(Beta[idx,:], K).reshape(-1,1))**K

        return mu

    def denoiser(self, q, r, d, tau):
        """
        Selects the proper denoiser to run as part of AMP
        :param q: prior probability
        :param r: effective observation
        :param d: signal amplitude
        :param tau: noise standard deviation
        """

        if self.__DenoiserType == 'PME':
            return self.pme(q, r, d, tau)

    def ampStateUpdate(self, z, s, d, Az, Kht, OuterCode):
        """
        Update state for a specific bin.
        :param z: AMP residual
        :param s: current AMP state
        :param d: amplitude of signal
        :param Az: transpose of the sensing matrix
        :param Kht: estimated number of users in the specified bin
        :param OuterCode: factor graph used as the outer code
        """

        # Initialize important parameters
        M = self.__M
        L = self.__L
        n = self.__n
        z = z.flatten()
        nbr = self.__numBlockRows
        numBPIter = self.__numBPIter

        # Effective observation and tau computations
        if self.__SensingMatrixType == 'Dense':
            tau = np.sqrt(np.sum(z**2)/n)
            tau = tau * np.ones(L)
            r = (d*s + Az(z))

        elif self.__SensingMatrixType == 'BlockDiagonal':
            r = np.zeros(s.shape)
            tau = np.zeros(L)
            for idxblock in range(L):
                tau[idxblock] = np.sqrt(np.sum(z[idxblock*nbr:(idxblock+1)*nbr]**2)/nbr)
                r[idxblock*M:(idxblock+1)*M] = d*s[idxblock*M:(idxblock+1)*M] + Az(z[idxblock*nbr:(idxblock+1)*nbr])

        # Compute priors
        mu = 0 if Kht == 0 else self.computePrior(r, OuterCode, Kht, tau, d)

        # Perform AMP denoising
        if self.__SensingMatrixType == 'Dense':
            s = self.denoiser(mu, r, d, tau[0])

        elif self.__SensingMatrixType == 'BlockDiagonal':
            for idxblock in range(L):
                s[idxblock*M:(idxblock+1)*M] = self.denoiser(mu[idxblock*M:(idxblock+1)*M], r[idxblock*M:(idxblock+1)*M], d, tau[idxblock])

        # return state update            
        return s

    def jointAMPResidual(self, y, z, sList, dmsg):
        """
        Compute residual for use within AMP iterate
        :param y: original observation
        :param z: AMP residual from the previous iteration
        :param sList: list of state updates through AMP composite iteration
        :param d: power allocated to each section
        """

        # Initialize important parameters
        n = self.__n
        L = self.__L
        M = self.__M
        nbr = self.__numBlockRows

        # Compute tau
        if self.__SensingMatrixType == 'Dense':
            tau = np.sqrt(np.sum(z**2)/n)
            tau = tau * np.ones(L)
        elif self.__SensingMatrixType == 'BlockDiagonal':
            tau = np.zeros(L)
            for idxblock in range(L):
                tau[idxblock] = np.sqrt(np.sum(z[idxblock*nbr:(idxblock+1)*nbr]**2)/nbr)
        print(tau)
        # Compute residual
        z_plus = y.copy()
        for idxbin in range(self.__numBins):

            if self.__SensingMatrixType == 'Dense':
                z_plus += -1*dmsg*self.__Ab[idxbin](sList[idxbin]) + (z/(n*tau[0]**2))* \
                           ((dmsg**2)*(np.sum(sList[idxbin]) - np.sum(sList[idxbin]**2)))
            
            elif self.__SensingMatrixType == 'BlockDiagonal':
                for idxblock in range(L):
                    z_plus[idxblock*nbr:(idxblock+1)*nbr] += -1*dmsg*self.__Ab[idxbin](sList[idxbin][idxblock*M:(idxblock+1)*M]) + \
                                                            (z[idxblock*nbr:(idxblock+1)*nbr]/(nbr*tau[idxblock]**2)) * \
                                                            ((dmsg**2)*(np.sum(sList[idxbin][idxblock*M:(idxblock+1)*M]) -  
                                                            np.sum(sList[idxbin][idxblock*M:(idxblock+1)*M]**2)))
        
        # return joint AMP residual
        return z_plus
