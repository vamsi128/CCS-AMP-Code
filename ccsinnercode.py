from pyfht import block_sub_fht
import numpy as np

class GenericInnerCode:
    """
    Class @class InnerCode creates an encoder/decoder for CCS using AMP with BP
    """

    def __init__(self, N, P, std, Ka, Graph):
        """
        Initialize encoder/decoder for CCS inner code
        :param N: total number of channel uses (real DOF)
        :param P: transmit power
        :param std: noise standard deviation
        :param Ka: number of active users
        :param Graph: outer graph
        """

        # Store Parameters
        self.__L = Graph.varcount
        self.__ml = Graph.sparseseclength
        self.__N = N
        self.__P = P
        self.__std = std
        self.__Ka = Ka
        self.__Phat = N*P/self.__L

    def getL(self):
        return self.__L
    
    def getML(self):
        return self.__ml
    
    def getN(self):
        return self.__N

    def getP(self):
        return self.__P

    def getPhat(self):
        return self.__Phat

    def getNumBlockRows(self):
        return self.__numBlockRows

    def getStd(self):
        return self.__std

    def getKa(self):
        return self.__Ka

    def SparcCodebook(self, L, ml, N):
        """
        Generate SPARC Codebook for CS encoding
        :param L: number of sections
        :param ml: length of each section
        :param N: number of channel uses (real DOF)
        """

        # generate Hadamard matrices 
        self.__numBlockRows = N
        self.__Ax, self.__Ay, _ = block_sub_fht(N, ml, L, ordering=None, seed=None)

    def Ab(self, b):
        return self.__Ax(b).reshape(-1, 1) / np.sqrt(self.__numBlockRows)

    def Az(self, z):
        return self.__Ay(z).reshape(-1, 1) / np.sqrt(self.__numBlockRows)

    def EncodeSection(self, x):
        """
        Compressed sensing encoding
        :param x: sparse vector to be CS encoded
        """
        return np.sqrt(self.__Phat) * self.Ab(x)

    def NoiseStdDeviation(self, z):
        """
        Compute noise standard deviation within the AMP iterate
        :param z: residual from previous iteration of AMP
        """
        return np.sqrt(np.sum(z**2)/len(z))

    def AmpDenoiser(self, q, s, tau):
        """
        Denoiser to be used within the AMP iterate
        :param q: vector of priors
        :param s: effective observation
        :param tau: standard deviation of noise
        """
        s = s.flatten()
        return ((q*np.exp(-(s-np.sqrt(self.__Phat))**2/(2*tau**2))) / \
                (q*np.exp(-(s-np.sqrt(self.__Phat))**2/(2*tau**2)) +  \
                (1-q)*np.exp(-s**2/(2*tau**2)))).astype(float).reshape(-1, 1)

    def ApproximateVector(self, x):

        # define k
        K = self.__Ka

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

    def ComputePrior(self, s, BPonOuterGraph, graph, tau, numBPIter):
        """
        Compute vector of priors within the AMP iterate
        :param s: effective observation
        :param BPonOuterGraph: indicates whether BP should be performed on the outer graph
        :param graph: outer graph
        :param tau: noise standard deviation
        :param numBPIter: number of BP iterations to perform
        """

        # Compute uninformative prior
        s = s.flatten()                             # force s to have the right shape
        p0 = 1-(1-1/self.__ml)**self.__Ka           # uninformative prior
        p1 = p0 * np.ones(s.shape, dtype=float)     # vector of uninformative priors

        # If not BPonOuterGraph, return uninformative prior
        if not BPonOuterGraph:
            return p1
        else:

            # Handle scalar and vector taus
            if np.isscalar(tau):
                tau = tau * np.ones(self.getL())

            # Prep for PME computation 
            localEstimates = np.zeros((self.__L, self.__ml), dtype=float)    # data structure for PME 
            m = self.getML()                                                 # use m as an alias for self.getML()

            # Translate the effective observation into a PME  
            for i in range(self.getL()): 
                localEstimates[i, :] = self.AmpDenoiser(p1[i*m:(i+1)*m], s[i*m:(i+1)*m], tau[i]).flatten()
                localEstimates[i, :] /= np.sum(localEstimates[i, :])

            # reset graph so that each message becomes uninformative
            graph.reset() 

            # set variable node observations - these are the lambdas
            for varnodeid in graph.varlist:
                graph.setobservation(varnodeid, localEstimates[varnodeid-1, :])

            # perform numBPIter of BP
            for idxit in range(numBPIter):
                graph.updatechecks()
                graph.updatevars()

            # Obtain belief vectors from the graph
            q = np.zeros((self.__L, self.__ml))
            for varnodeid in graph.varlist:
                extrinsicprior = self.ApproximateVector(graph.getextrinsicestimate(varnodeid))
                q[varnodeid-1,:] = 1 - (1 - extrinsicprior)**self.__Ka

            return np.minimum(q.flatten(), 1)

    def EffectiveObservation(self, xHt, z):
        """
        Effective observation for AMP
        :param xHt: estimate of vector x
        :param z: AMP residual
        """

        return (np.sqrt(self.__Phat)*xHt + self.Az(z.flatten())).astype(np.longdouble)

    def Residual(self, xHt, y, z, tau):
        """
        Compute residual during AMP iterate
        :param xHt: estimate of vector x
        :param y: vector of observations of x
        :param z: previous residual
        :param tau: noise standard deviation
        """

        return y - np.sqrt(self.__Phat)*self.Ab(xHt) + (z/(self.__numBlockRows*tau**2)) * \
                    (self.__Phat*np.sum(xHt) - self.__Phat*np.sum(xHt**2))             # compute residual

class DenseInnerCode(GenericInnerCode):
    """
    Class @class DenseInnerCode creates a CS encoder/decoder using a dense sensing matrix
    """

    def __init__(self, N, P, std, Ka, Graph):
        """
        Initialize encoder/decoder for CCS inner code
        :param N: total number of channel uses (real DOF)
        :param P: transmit power
        :param std: noise standard deviation
        :param Ka: number of active users
        :param Graph: outer graph
        """
        super().__init__(N, P, std, Ka, Graph)

        # Create dense sensing matrix A
        self.SparcCodebook(self.getL(), self.getML(), N)

    def Encode(self, x):
        """
        Encode signal using dense sensing matrix A
        :param x: signal to encode
        """
        return super().EncodeSection(x)

    def Decode(self, y, numAmpIter, BPonOuterGraph=False, numBPIter=1, graph=None):
        """
        AMP for support recovery of x given observation y
        :param y: observations of x
        :param numAmpIter: number of iterations of AMP to perform
        :param BPonOuterGraph: whether BP should be performed on outer graph.  Default = false
        :param numBPIter: number of BP iterations to be performed.  Default = 1
        :param graph: graphical structure of outer code.  Default = None
        """
  
        xHt = np.zeros((self.getL()*self.getML(), 1)) # data structure to store support of x
        z = y.copy()                                  # deep copy of y for AMP to modify
        tauEvolution = np.zeros((numAmpIter, 1))      # track how tau changes with each iteration
        
        # perform numAmpIter iterations of AMP
        for t in range(numAmpIter):

            tau = self.NoiseStdDeviation(z)                  # compute std of noise using residual
            s = self.EffectiveObservation(xHt, z)            # effective observation
            q = self.ComputePrior(s, BPonOuterGraph, graph, tau, numBPIter).flatten()
            xHt = self.AmpDenoiser(q, s, tau)                # run estimate through denoiser
            z = self.Residual(xHt, y, z, tau)                # compute residual           
            tauEvolution[t] = tau                            # store tau

        return xHt, tauEvolution

class BlockDiagonalInnerCode(GenericInnerCode):
    """
    Class @class BlockDiagonalInnerCode creates a CS encoder/decoder using a block 
    diagonal sensing matrix
    """

    def __init__(self, N, P, std, Ka, Graph):
        """
        Initialize encoder/decoder for CCS inner code
        :param N: total number of channel uses (real DOF)
        :param P: transmit power
        :param std: noise standard deviation
        :param Ka: number of active users
        :param Graph: outer graph
        """
        super().__init__(N, P, std, Ka, Graph)

        # Determine number of rows per block in A
        assert N % self.getL() == 0, "N must be a multiple of L"
        self.__numBlockRows = N // self.getL()

        # Create block of A
        self.SparcCodebook(1, self.getML(), self.__numBlockRows)

    def Encode(self, x):
        """
        Encode signal using block diagonal sensing matrix A
        :param Phat: estimated power
        :param x: signal to encode
        """

        # instantiate data structure for y
        y = np.zeros(self.getN())

        # encode each section individually
        for i in range(self.getL()):
            y[i*self.__numBlockRows:(i+1)*self.__numBlockRows] = self.EncodeSection(x[i*self.getML():(i+1)*self.getML()]).flatten()

        # return encoded signal y
        return y.reshape(-1, 1)

    def Decode(self, y, numAmpIter, BPonOuterGraph=False, numBPIter=1, graph=None):
        """
        AMP for support recovery of x given observation y
        :param y: observations of x
        :param numAmpIter: number of iterations of AMP to perform
        :param BPonOuterGraph: whether BP should be performed on outer graph.  Default = false
        :param numBPIter: number of BP iterations to be performed.  Default = 1
        :param graph: graphical structure of outer code.  Default = None
        """

        xHt = np.zeros((self.getL()*self.getML(), 1)) # data structure to store support of x
        s = np.zeros((self.getL()*self.getML(), 1))   # data structure to store effective observations
        z = y.copy()                                  # deep copy of y for AMP to modify
        tau = np.zeros((self.getL(), 1))              # data structure to store noise standard deviations
        tauEvolution = np.zeros((numAmpIter, 1))      # track how tau changes with each iteration
        n = self.__numBlockRows                       # use n as an alias for self.__numBlockRows
        m = self.getML()                              # length of each section

        # Perform numAmpIter iterations of AMP
        for t in range(numAmpIter):

            # Iterate through each of the L sections
            for i in range(self.getL()):
                tau[i] = self.NoiseStdDeviation(z[i*n:(i+1)*n])                                    # compute noise std dev
                s[i*m:(i+1)*m] = self.EffectiveObservation(xHt[i*m:(i+1)*m], z[i*n:(i+1)*n])       # effective observation

            # Compute priors
            q = self.ComputePrior(s, BPonOuterGraph, graph, tau, numBPIter)                        # vector of priors
            
            # Iterate through each of the L sections
            for i in range(self.getL()):
                xHt[i*m:(i+1)*m] = self.AmpDenoiser(q[i*m:(i+1)*m], s[i*m:(i+1)*m], tau[i])        # apply denoiser
                z[i*n:(i+1)*n] = self.Residual(xHt[i*m:(i+1)*m], y[i*n:(i+1)*n], z[i*n:(i+1)*n], \
                                               tau[i])                                             # compute residual 
            
            # Track evolution of noise standard deviation vs iteration
            tauEvolution[t] = tau[0]                           

        return xHt, tauEvolution

class CodedDemixingInnerCode(GenericInnerCode):
    """
    Class @class CodedDemixingInnerCode creates a CS encoder/decoder that facilitates 
    multiple classes of users.  
    """

    def __init__(self, N, P, std, K, numBins, BlockDiagonal, OuterCodes):
        """
        Initialize encoder/decoder for CCS inner code
        :param N: total number of channel uses (real DOF)
        :param P: transmit power
        :param std: noise standard deviation
        :param K: number of active users per bin
        :param numBins: number of bins/classes employed in simulation
        :param BlockDiagonal: boolean flag indicates whether to use block diagonal matrices
        :param OuterCodes: list of outer factor graphs
        """

        # Store important parameters
        self.__P = P if not np.isscalar(P) else P*np.ones(numBins)
        self.__N = N
        self.__OuterCodes = OuterCodes
        self.__numBins = numBins
        self.__BlockDiagonal = BlockDiagonal
        self.__K = K.copy()
        self.__InnerCodes = [
            BlockDiagonalInnerCode(N, self.__P[i], std, K[i], OuterCodes[i]) if BlockDiagonal else 
            DenseInnerCode(N, self.__P[i], std, K[i], OuterCodes[i])
            for i in range(numBins)
        ]

    def JointAMPResidual(self, y, z, s, tau):
        """
        Jointly compute AMP residual
        :param y: original observation
        :param z: AMP residual from the previous iteration
        :param s: list of state updates for each bin
        :param tau: standard deviation of effective observation noise
        """

        # create deep copy of y
        z_plus = y.copy()

        # iterate through each bin
        for i in range(self.__numBins):

            # get power associated with specific class
            P = self.__InnerCodes[i].getPhat()

            # compute estimation error
            if self.__BlockDiagonal:
                nbr = self.__InnerCodes[i].getNumBlockRows()
                M = self.__InnerCodes[i].getML()
                estimationError = np.zeros(y.shape)
                onsagerTerm = np.zeros(y.shape)

                for j in range(self.__InnerCodes[i].getL()):
                    estimationError[j*nbr:(j+1)*nbr] = np.sqrt(P)*self.__InnerCodes[i].Ab(s[i][j*M:(j+1)*M])
                    onsagerTerm[j*nbr:(j+1)*nbr] = (z[j*nbr:(j+1)*nbr]/(nbr*tau[j]**2))*(P*(np.sum(s[i][j*M:(j+1)*M]) - np.sum(s[i][j*M:(j+1)*M]**2)))

            else:
                estimationError = np.sqrt(P)*self.__InnerCodes[i].Ab(s[i])
                N = self.__InnerCodes[i].getNumBlockRows()
                onsagerTerm = (z/(N*tau**2))*(P*(np.sum(s[i]) - np.sum(s[i]**2)))

            # adjust z_plus with Onsager correction term + estimate error
            z_plus += (-1*estimationError) + onsagerTerm                      

        # return joint AMP residual
        return z_plus

    def Encode(self, signals):
        """
        Encode signal using dense sensing matrices. 
        :param signals: user codewords to encode grouped by bin
        """

        # add contribution from each class
        x = 0.0
        for i in range(self.__numBins):
            x += self.__InnerCodes[i].Encode(signals[i])

        # return encoded signal
        return x

    def Decode(self, y, numAmpIter, BPonOuterGraph, numBPIter=1):
        """
        AMP for support recovery of x given observation y
        :param y: observations of x
        :param numAmpIter: number of iterations of AMP to perform
        :param BPonOuterGraph: whether BP should be performed on outer graph.  Default = false
        :param numBPIter: number of BP iterations to be performed.  Default = 1
        """

        # prepare for AMP decoding
        z = y.copy()
        s = [ 
            np.zeros((self.__InnerCodes[i].getL()*self.__InnerCodes[i].getML(), 1))
            for i in range(self.__numBins)
        ]

        # AMP decoding
        for idxampiter in range(numAmpIter):

            # Update state estimate of each bin individually
            for i in range(self.__numBins):
                
                # compute effective observations
                if self.__BlockDiagonal:
                    L, M, NBR = self.__InnerCodes[i].getL(), self.__InnerCodes[i].getML(), self.__InnerCodes[i].getNumBlockRows()
                    tau = np.zeros(L)

                    # block-wise effective observation
                    for j in range(L):
                        s[i][j*M:(j+1)*M] = self.__InnerCodes[i].EffectiveObservation(s[i][j*M:(j+1)*M], z[j*NBR:(j+1)*NBR])
                        tau[j] = np.sqrt(np.sum(z[j*NBR:(j+1)*NBR]**2)/len(z[j*NBR:(j+1)*NBR]))

                    # compute priors
                    q = 0 if self.__K[i] == 0 else self.__InnerCodes[i].ComputePrior(s[i], BPonOuterGraph, self.__OuterCodes[i], tau, numBPIter).flatten()

                    # block-wise AMP denoiser
                    for j in range(L):
                        s[i][j*M:(j+1)*M] = self.__InnerCodes[i].AmpDenoiser(q[j*M:(j+1)*M], s[i][j*M:(j+1)*M], tau[j])
                else:
                    # effective observation + compute tau
                    s[i] = self.__InnerCodes[i].EffectiveObservation(s[i], z)
                    tau = np.sqrt(np.sum(z**2)/len(z))
                    
                    # compute priors
                    q = 0 if self.__K[i] == 0 else self.__InnerCodes[i].ComputePrior(s[i], BPonOuterGraph, self.__OuterCodes[i], tau, numBPIter).flatten()

                    # AMP denoiser
                    s[i] = self.__InnerCodes[i].AmpDenoiser(q, s[i], tau)               

            # Compute residual jointly
            z = self.JointAMPResidual(y, z, s, tau)

        # return result
        return s
