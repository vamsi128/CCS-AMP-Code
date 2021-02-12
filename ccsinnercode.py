from pyfht import block_sub_fht
import numpy as np

class GenericInnerCode:

    """
    Class @class InnerCode creates an encoder/decoder for CCS using AMP with BP
    """

    def __init__(self, L, ml, N, P, std, Ka):
        """
        Initialize encoder/decoder for CCS inner code
        :param L: Number of sections/sub-blocks
        :param ml: Length of each section of the message
        :param N: total number of channel uses (real DOF)
        :param P: transmit power
        :param std: noise standard deviation
        :param Ka: number of active users
        """

        # Store Parameters
        self.__L = L
        self.__ml = ml
        self.__N = N
        self.__P = P
        self.__std = std
        self.__Ka = Ka

    def SparcCodebook(self, L, ml, N):
        """
        Generate SPARC Codebook for CS encoding
        :param L: number of sections
        :param ml: length of each section
        :param N: number of channel uses (real DOF)
        """
        self.__Ax, self.__Ay, _ = block_sub_fht(N, ml, L, ordering=None, seed=None)

    def Ab(self, b):
        return self.__Ax(b).reshape(-1, 1) / np.sqrt(self.__N)

    def Az(self, z):
        return self.__Ay(z).reshape(-1, 1) / np.sqrt(self.__N)

    def EncodeSection(self, Phat, x):
        """
        Compressed sensing encoding
        :param Phat: estimated power
        :param x: sparse vector to be CS encoded
        """
        return np.sqrt(Phat) * self.Ab(x)

    def NoiseStdDeviation(self, z):
        """
        Compute noise standard deviation within the AMP iterate
        :param z: residual from previous iteration of AMP
        """
        return np.sqrt(np.sum(z**2)/len(z))

    def AmpDenoiser(self, q, s, tau, Phat):
        """
        Denoiser to be used within the AMP iterate
        :param q: vector of priors
        :param s: effective observation
        :param tau: standard deviation of noise
        :param Phat: estimate of codeword power
        """

        return (q*np.exp(-(s-np.sqrt(Phat))**2/(2*tau**2)))/ \
               (q*np.exp(-(s-np.sqrt(Phat))**2/(2*tau**2)) + \
               (1-q)*np.exp(-s**2/(2*tau**2))).astype(float).reshape(-1, 1)

    def ComputePrior(self, s, BPonOuterGraph, graph, tau, Phat, numBPIter):
        """
        Compute vector of priors within the AMP iterate
        :param s: effective observation
        :param BPonOuterGraph: indicates whether BP should be performed on the outer graph
        :param graph: outer graph
        :param tau: noise standard deviation
        :param Phat: estimate of codeword power
        :param numBPIter: number of BP iterations to perform
        """

        # Compute uninformative prior
        p0 = 1-(1-1/self.__ml)**self.__Ka

        # If not BPonOuterGraph, return uninformative prior
        if not BPonOuterGraph:
            return p0
        else:

            # Translate the effective observation into a PME
            p1 = p0 * np.ones(s.shape, dtype=float)
            pme = (p1*np.exp(-(s-np.sqrt(Phat))**2/(2*tau**2)))/ \
                  (p1*np.exp(-(s-np.sqrt(Phat))**2/(2*tau**2)) + \
                  (1-p1)*np.exp(-s**2/(2*tau**2))).astype(float).reshape(-1, 1)
            pme = pme.reshape(self.__L,-1)                          # Reshape PME into an LxM matrix
            pme = pme/(np.sum(pme,axis=1).reshape(self.__L,-1))     # normalize rows of PME

            # reset graph so that each message becomes all 1s
            graph.reset() 

            # set variable node observations - these are the lambdas
            for i in range(self.__L):
                graph.setobservation(i, pme[i,:])

            # perform numBPIter of BP
            for idxit in range(numBPIter):
                graph.updatevars()

            # Obtain belief vectors from the graph
            return graph.getestimates().flatten()

    def EffectiveObservation(self, Phat, xHt, z):
        """
        Effective observation for AMP
        :param Phat: estimate of codeword power
        :param xHt: estimate of vector x
        :param z: AMP residual
        """

        return (np.sqrt(Phat)*xHt + self.Az(z)).astype(np.longdouble)

    def Residual(self, xHt, y, z, Phat, tau):
        """
        Compute residual during AMP iterate
        :param xHt: estimate of vector x
        :param y: vector of observations of x
        :param z: previous residual
        :param Phat: estimate of codeword power
        :param tau: noise standard deviation
        """

        return y - np.sqrt(Phat)*self.Ab(xHt) + (z/(self.__N*tau**2)) * \
                    (Phat*np.sum(xHt) - Phat*np.sum(xHt**2))             # compute residual


class DenseInnerCode(GenericInnerCode):
    """
    Class @class DenseInnerCode creates a CS encoder/decoder using a dense sensing matrix
    """

    def __init__(self, L, ml, N, P, std, Ka):
        """
        Initialize encoder/decoder for CCS inner code
        :param L: Number of sections/sub-blocks
        :param ml: Length of each section of the message
        :param N: total number of channel uses (real DOF)
        :param P: transmit power
        :param std: noise standard deviation
        :param Ka: number of active users
        """
        super().__init__(L, ml, N, P, std, Ka)

        # Create dense sensing matrix A
        self.SparcCodebook(L, ml, N)

    def Encode(self, Phat, x):
        """
        Encode signal using dense sensing matrix A
        :param Phat: estimated power
        :param x: signal to encode
        """
        return self.EncodeSection(Phat, x)

    def Decode(self, y, numAmpIter, BPonOuterGraph=False, numBPIter=1, graph=None):
        """
        AMP for support recovery of x given observation y
        :param y: observations of x
        :param numAmpIter: number of iterations of AMP to perform
        :param BPonOuterGraph: whether BP should be performed on outer graph.  Default = false
        :param numBPIter: number of BP iterations to be performed.  Default = 1
        :param graph: graphical structure of outer code.  Default = None
        """

        xHt = np.zeros((self.__L*self.__ml, 1))       # data structure to store support of x
        z = y.copy()                                  # deep copy of y for AMP to modify
        Phat = self.__N*self.__P/self.__L             # Estimate of transmit power
        tauEvolution = np.zeros((numAmpIter, 1))      # track how tau changes with each iteration

        # perform numAmpIter iterations of AMP
        for t in range(numAmpIter):

            tau = self.NoiseStdDeviation(z)                  # compute std of noise using residual
            s = self.EffectiveObservation(Phat, xHt, z)      # effective observation
            q = self.ComputePrior(s, BPonOuterGraph, graph, tau, Phat, numBPIter) 
            xHt = self.AmpDenoiser(q, s, tau, Phat)          # run estimate through denoiser
            z = self.Residual(xHt, y, z, Phat, tau)          # compute residual           
            tauEvolution[t] = tau                            # store tau

        return xHt, tauEvolution


class BlockDiagonalInnerCode(GenericInnerCode):
    """
    Class @class BlockDiagonalInnerCode creates a CS encoder/decoder using a block 
    diagonal sensing matrix
    """

    def __init__(self, L, ml, N, P, std, Ka):
        """
        Initialize encoder/decoder for CCS inner code
        :param L: Number of sections/sub-blocks
        :param ml: Length of each section of the message
        :param N: total number of channel uses (real DOF)
        :param P: transmit power
        :param std: noise standard deviation
        :param Ka: number of active users
        """
        super().__init__(L, ml, N, P, std, Ka)

        # Determine number of rows per block in A
        assert N % L == 0, "N must be a multiple of L"
        self.__numBlockRows = N // L

        # Create block of A
        self.SparcCodebook(1, ml, self.__numBlockRows)

    def Encode(self, Phat, x):
        """
        Encode signal using block diagonal sensing matrix A
        :param Phat: estimated power
        :param x: signal to encode
        """

        # instantiate data structure for y
        y = np.zeros(x.shape)

        # encode each section individually
        for i in range(self.__L):
            y[i*self.__numBlockRows:(i+1)*self.__numBlockRows] = self.EncodeSection(Phat, x[i*self.__ml:(i+1)*self.__ml])

        # return encoded signal y
        return y

    def Decode(self, y, numAmpIter, BPonOuterGraph=False, numBPIter=1, graph=None):
        """
        AMP for support recovery of x given observation y
        :param y: observations of x
        :param numAmpIter: number of iterations of AMP to perform
        :param BPonOuterGraph: whether BP should be performed on outer graph.  Default = false
        :param numBPIter: number of BP iterations to be performed.  Default = 1
        :param graph: graphical structure of outer code.  Default = None
        """

        xHt = np.zeros((self.__L*self.__ml, 1))       # data structure to store support of x
        s = np.zeros((self.__L*self.__ml, 1))         # data structure to store effective observations
        z = y.copy()                                  # deep copy of y for AMP to modify
        Phat = self.__N*self.__P/self.__L             # Estimate of transmit power
        tau = np.zeros((L, 1))                        # data structure to store noise standard deviations
        tauEvolution = np.zeros((numAmpIter, 1))      # track how tau changes with each iteration
        n = self.__numBlockRows                       # use n as an alias for self.__numBlockRows
        m = self.__ml                                 # use m as an alias for self.__ml      

        # Perform numAmpIter iterations of AMP
        for t in range(numAmpIter):

            # Iterate through each of the L sections
            for i in range(self.__L):
                tau[i] = self.NoiseStdDeviation(z[i*n:(i+1)*n])                                    # compute noise std dev
                s[i*m:(i+1)*m] = self.EffectiveObservation(Phat, xHt[i*m:(i+1)*m], z[i*n:(i+1)*n]) # effective observation

            # Compute priors
            q = self.ComputePrior(s, BPonOuterGraph, graph, tau, Phat, numBPIter)                  # vector of priors
            
            # Iterate through each of the L sections
            for i in range(self.__L):
                xHt[i*m:(i+1)*m] = self.AmpDenoiser(q[i*m:(i+1)*m], s[i*m:(i+1)*m], tau[i], Phat)  # apply denoiser
                z[i*n:(i+1)*n] = self.Residual(xHt[i*m:(i+1)*m], y[i*n:(i+1)*n], z[i*n:(i+1)*n], \
                                               Phat, tau[i])                                       # compute residual 
            
            # Track evolution of noise standard deviation vs iteration
            tauEvolution[t] = tau[0]                           

        return xHt, tauEvolution