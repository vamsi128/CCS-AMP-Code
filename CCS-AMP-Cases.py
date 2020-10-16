import numpy as np
import ipdb
import matplotlib.pyplot as plt
import time
from pyfht import block_sub_fht

def Tree_encode(tx_message,K,messageBlocks,G,L,J):
    encoded_tx_message = np.zeros((K,L),dtype=int)
    
    encoded_tx_message[:,0] = tx_message[:,0:J].dot(2**np.arange(J)[::-1])
    for i in range(1,L):
        if messageBlocks[i]:
            # copy the message if i is an information section
            encoded_tx_message[:,i] = tx_message[:,np.sum(messageBlocks[:i])*J:(np.sum(messageBlocks[:i])+1)*J].dot(2**np.arange(J)[::-1])
        else:
            # compute the parity if i is a parity section
            indices = np.where(G[i])[0]
            ParityInteger=np.zeros((K,1),dtype='int')
            for j in indices:
                ParityInteger1 = encoded_tx_message[:,j].reshape(-1,1)
                ParityInteger = np.mod(ParityInteger+ParityInteger1,2**J)
            encoded_tx_message[:,i] = ParityInteger.reshape(-1)
    
    return encoded_tx_message

def convert_indices_to_sparse(encoded_tx_message_indices,L,J,K):
    encoded_tx_message_sparse=np.zeros((L*2**J,1),dtype=int)
    for i in range(L):
        A = encoded_tx_message_indices[:,i]
        B = A.reshape([-1,1])
        np.add.at(encoded_tx_message_sparse, i*2**J+B, 1)        
    return encoded_tx_message_sparse

def convert_sparse_to_indices(cs_decoded_tx_message_sparse,L,J,listSize):
    cs_decoded_tx_message = np.zeros((listSize,L),dtype=int)
    for i in range(L):
        A = cs_decoded_tx_message_sparse[i*2**J:(i+1)*2**J]
        idx = (A.reshape(2**J,)).argsort()[np.arange(2**J-listSize)]
        B = np.setdiff1d(np.arange(2**J),idx)
        cs_decoded_tx_message[:,i] = B 

    return cs_decoded_tx_message

def extract_msg_indices(Paths,cs_decoded_tx_message, L,J):
    msg_bits = np.empty(shape=(0,0))
    L1 = Paths.shape[0]
    for i in range(L1):
        msg_bit=np.empty(shape=(0,0))
        path = Paths[i].reshape(1,-1)
        for j in range(path.shape[1]):
            msg_bit = np.hstack((msg_bit,cs_decoded_tx_message[path[0,j],j].reshape(1,-1))) if msg_bit.size else cs_decoded_tx_message[path[0,j],j]
            msg_bit=msg_bit.reshape(1,-1)
        msg_bits = np.vstack((msg_bits,msg_bit)) if msg_bits.size else msg_bit           
    return msg_bits

def sparc_codebook(L, M, n,P):
    Ax, Ay, _ = block_sub_fht(n, M, L, ordering=None)
    def Ab(b):
        return Ax(b).reshape(-1, 1)/ np.sqrt(n)
    def Az(z):
        return Ay(z).reshape(-1, 1)/ np.sqrt(n) 
    return Ab, Az

def computePrior(s,G,messageBlocks,L,M,p0,K,tau,Phat,numBPiter,case):
    
    q = np.zeros(s.shape,dtype=float)
    p1 = p0*np.ones(s.shape,dtype=float)
    temp_beta = np.zeros((L*M, 1))
    
    for iter in range(numBPiter):
        
        # Translate the effective observation into PME. For the first iteration of BP, use the uninformative prior p0
        if case==1:
            for i in range(L):
                temp_beta[i*M:(i+1)*M] = (p1[i*M:(i+1)*M]*np.exp(-(s[i*M:(i+1)*M]-np.sqrt(Phat))**2/(2*tau[i]**2)))/ \
                                         (p1[i*M:(i+1)*M]*np.exp(-(s[i*M:(i+1)*M]-np.sqrt(Phat))**2/(2*tau[i]**2)) + \
                                         (1-p1[i*M:(i+1)*M])*np.exp(-s[i*M:(i+1)*M]**2/(2*tau[i]**2))).astype(float) \
                                         .reshape(-1, 1)
        else:
            temp_beta = (p1*np.exp(-(s-np.sqrt(Phat))**2/(2*tau**2)))/ (p1*np.exp(-(s-np.sqrt(Phat))**2/(2*tau**2)) + (1-p1)*np.exp(-s**2/(2*tau**2))).astype(float).reshape(-1, 1)

    
        # Reshape PME into an LxM matrix
        Beta = temp_beta.reshape(L,-1)
        #print(Beta.shape,np.sum(Beta,axis=1))
        Beta = Beta/(np.sum(Beta,axis=1).reshape(L,-1))
        # Rotate PME 180deg about y-axis
        Betaflipped = np.hstack((Beta[:,0].reshape(-1,1),np.flip(Beta[:,1:],axis=1)))
        # Compute and store all FFTs
        BetaFFT = np.fft.fft(Beta)
        BetaflippedFFT = np.fft.fft(Betaflipped)
        for i in range(L):
            if messageBlocks[i]:
                # Parity sections connected to info section i
                parityIndices = np.where(G[i])[0]
                BetaIFFTprime = np.empty((0,0)).astype(float)
                for j in parityIndices:
                    # Other info blocks connected to this parity block
                    messageIndices = np.setdiff1d(np.where(G[j])[0],i)
                    BetaFFTprime = np.vstack((BetaFFT[j],BetaflippedFFT[messageIndices,:]))
                    # Multiply the relevant FFTs
                    BetaFFTprime = np.prod(BetaFFTprime,axis=0)
                    # IFFT
                    BetaIFFTprime1 = np.fft.ifft(BetaFFTprime).real
                    BetaIFFTprime = np.vstack((BetaIFFTprime,BetaIFFTprime1)) if BetaIFFTprime.size else BetaIFFTprime1
                BetaIFFTprime = np.prod(BetaIFFTprime,axis=0)
            else:
                BetaIFFTprime = np.empty((0,0)).astype(float)
                # Information sections connected to this parity section (assuming no parity over parity sections)
                Indices = np.where(G[i])[0]
                # FFT
                BetaFFTprime = BetaFFT[Indices,:]
                # Multiply the relevant FFTs
                BetaFFTprime = np.prod(BetaFFTprime,axis=0)
                # IFFT
                BetaIFFTprime = np.fft.ifft(BetaFFTprime).real
            
            # Normalize to ensure it sums to one
            p1[i*M:(i+1)*M] = (BetaIFFTprime/np.sum(BetaIFFTprime)).reshape(-1,1)
            p1[i*M:(i+1)*M]  = 1-(1-p1[i*M:(i+1)*M] )**K 
            # Normalize to ensure sum of priors within a section is K (optional)
            #p1[i*M:(i+1)*M] = p1[i*M:(i+1)*M]*K/np.sum(p1[i*M:(i+1)*M])
         
    q = np.minimum(p1,1)          
    return q

def amp(y, sigma_n, P, L, M, T, Ab, Az,p0,K,G,messageBlocks,BPonOuterGraph,numBPiter,case):

    # set up AMP parameters
    n = y.size
    Beta = np.zeros((L*M, 1))
    s = np.zeros((L*M, 1))
    z = y.copy()
    Phat = n*P/L
    tau_evolution = np.zeros((T,1))
    
    # set up case-specific parameters
    if case != 2:
        tau = np.zeros((L, 1))
    
    # begin AMP iterations
    for t in range(T):
        
        # Begin case-specific code
        
        # CCS (L independent instances of AMP)
        if case==0:
            # set up case-0 specific parameters
            numBlockRows = n//L
            
            # iterate through each of L independent instances of AMP
            for i in range(L):
                # Compute tau online using the residual
                tau[i] = np.sqrt(np.sum(z[i*numBlockRows:(i+1)*numBlockRows]**2)/numBlockRows)
            
                # compute effective observation
                Azz = Az(z[i*numBlockRows:(i+1)*numBlockRows]).astype(np.longdouble)
                s[i*M:(i+1)*M] = (np.sqrt(Phat)*Beta[i*M:(i+1)*M] + Azz).astype(np.longdouble)
                
                # use uninformative prior
                q = p0
                
                # denoiser
                Beta[i*M:(i+1)*M] = (q*np.exp(-(s[i*M:(i+1)*M]-np.sqrt(Phat))**2/(2*tau[i]**2)))/ \
                                 (q*np.exp(-(s[i*M:(i+1)*M]-np.sqrt(Phat))**2/(2*tau[i]**2)) + (1-q)*np.exp(-s[i*M:(i+1)*M]**2/(2*tau[i]**2))) \
                                 .astype(float).reshape(-1, 1)
                
                # residual
                z[i*numBlockRows:(i+1)*numBlockRows] = y[i*numBlockRows:(i+1)*numBlockRows] - np.sqrt(Phat)*Ab(Beta[i*M:(i+1)*M]) + \
                                                       (z[i*numBlockRows:(i+1)*numBlockRows]/(numBlockRows*tau[i]**2)) * \
                                                       (Phat*np.sum(Beta[i*M:(i+1)*M]) - Phat*np.sum(Beta[i*M:(i+1)*M]**2))
            # store value for tau_evolution
            tau_evolution[t] = tau[0]
            
            # end of case-0 specific code
            
        # CCS-Hybrid
        elif case==1:
            # set up case-1 specific parameters
            numBlockRows = n//L
            
            # iterate through each of L independent instances of AMP
            for i in range(L):
                # Compute tau online using the residual
                tau[i] = np.sqrt(np.sum(z[i*numBlockRows:(i+1)*numBlockRows]**2)/numBlockRows)
            
                # compute effective observation
                Azz = Az(z[i*numBlockRows:(i+1)*numBlockRows]).astype(np.longdouble)
                s[i*M:(i+1)*M] = (np.sqrt(Phat)*Beta[i*M:(i+1)*M] + Azz).astype(np.longdouble)
                
            # compute priors tying together all L instances of AMP
            q = computePrior(s,G,messageBlocks,L,M,p0,K,tau,Phat,numBPiter,case)
                
            # iterate through each of L independent instances of AMP
            for i in range(L):
                # denoiser
                Beta[i*M:(i+1)*M] = (q[i*M:(i+1)*M]*np.exp(-(s[i*M:(i+1)*M]-np.sqrt(Phat))**2/(2*tau[i]**2)))/ \
                                 (q[i*M:(i+1)*M]*np.exp(-(s[i*M:(i+1)*M]-np.sqrt(Phat))**2/(2*tau[i]**2)) + \
                                 (1-q[i*M:(i+1)*M])*np.exp(-s[i*M:(i+1)*M]**2/(2*tau[i]**2))).astype(float).reshape(-1, 1)
                
                # residual
                z[i*numBlockRows:(i+1)*numBlockRows] = y[i*numBlockRows:(i+1)*numBlockRows] - np.sqrt(Phat)*Ab(Beta[i*M:(i+1)*M]) + \
                                                       (z[i*numBlockRows:(i+1)*numBlockRows]/(numBlockRows*tau[i]**2)) * \
                                                       (Phat*np.sum(Beta[i*M:(i+1)*M]) - Phat*np.sum(Beta[i*M:(i+1)*M]**2))
            # store value for tau_evolution
            tau_evolution[t] = tau[0]
            
            # end of case-1 specific code
            
        # CCS-AMP with and without BP on outer graph
        elif case==2 or case==3:
            # Compute tau online using the residual
            tau = np.sqrt(np.sum(z**2)/n)

            # effective observation
            s = (np.sqrt(Phat)*Beta + Az(z)).astype(np.longdouble)

            if BPonOuterGraph==0:
                # Use the uninformative prior p0 for Giuseppe's scheme
                q = p0
            else:
                # Compute the prior through BP on outer graph
                q = computePrior(s,G,messageBlocks,L,M,p0,K,tau,Phat,numBPiter,case)

            # denoiser
            Beta = (q*np.exp(-(s-np.sqrt(Phat))**2/(2*tau**2)))/ (q*np.exp(-(s-np.sqrt(Phat))**2/(2*tau**2)) + (1-q)*np.exp(-s**2/(2*tau**2))).astype(float).reshape(-1, 1)

            # residual
            z = y - np.sqrt(Phat)*Ab(Beta) + (z/(n*tau**2)) * (Phat*np.sum(Beta) - Phat*np.sum(Beta**2))
            
            # update tau_evolution
            tau_evolution[t] = tau
            
            # end of case-2 specific code
        
        # End case-specific code
        

    return Beta, tau_evolution

def Tree_decoder(cs_decoded_tx_message,G,L,J,B,listSize):
    
    tree_decoded_tx_message = np.empty(shape=(0,0))
    
    Paths012 = merge_paths(cs_decoded_tx_message[:,0:3])
    
    Paths345 = merge_paths(cs_decoded_tx_message[:,3:6])
    
    Paths678 = merge_paths(cs_decoded_tx_message[:,6:9])
    
    Paths91011 = merge_paths(cs_decoded_tx_message[:,9:12])
    
    Paths01267812 = merge_pathslevel2(Paths012,Paths678,cs_decoded_tx_message[:,[0,6,12]])
    
    Paths3459101113 = merge_pathslevel2(Paths345,Paths91011,cs_decoded_tx_message[:,[3,9,13]])
    
    Paths01267812345910111314 = merge_all_paths0(Paths01267812,Paths3459101113,cs_decoded_tx_message[:,[1,4,10,14]])
    
    Paths = merge_all_paths_final(Paths01267812345910111314,cs_decoded_tx_message[:,[7,10,15]])
    
    
   
    return Paths

def merge_paths(A):
    listSize = A.shape[0]
    B = np.array([np.mod(A[:,0] + a,2**16) for a in A[:,1]]).flatten()
     
    Paths=np.empty((0,0))
    
    for i in range(listSize):
        I = np.where(B==A[i,2])[0].reshape(-1,1)
        if I.size:
            I1 = np.hstack([np.mod(I,listSize).reshape(-1,1),np.floor(I/listSize).reshape(-1,1)]).astype(int)
            Paths = np.vstack((Paths,np.hstack([I1,np.repeat(i,I.shape[0]).reshape(-1,1)]))) if Paths.size else np.hstack([I1,np.repeat(i,I.shape[0]).reshape(-1,1)])
    
    return Paths

def merge_pathslevel2(Paths012,Paths678,A):
    listSize = A.shape[0]
    Paths0 = Paths012[:,0]
    Paths6 = Paths678[:,0]
    B = np.array([np.mod(A[Paths0,0] + a,2**16) for a in A[Paths6,1]]).flatten()
    
    Paths=np.empty((0,0))
    
    for i in range(listSize):
        I = np.where(B==A[i,2])[0].reshape(-1,1)
        if I.size:
            I1 = np.hstack([np.mod(I,Paths0.shape[0]).reshape(-1,1),np.floor(I/Paths0.shape[0]).reshape(-1,1)]).astype(int)
            PPaths = np.hstack((Paths012[I1[:,0]].reshape(-1,3),Paths678[I1[:,1]].reshape(-1,3),np.repeat(i,I1.shape[0]).reshape(-1,1)))
            Paths = np.vstack((Paths,PPaths)) if Paths.size else PPaths
               
    return Paths


def merge_all_paths0(Paths01267812,Paths3459101113,A):
    listSize = A.shape[0]
    Paths1 = Paths01267812[:,1]
    Paths4 = Paths3459101113[:,1]
    Paths10 = Paths3459101113[:,4]
    Aa = np.mod(A[Paths4,1]+A[Paths10,2],2**16)
    B = np.array([np.mod(A[Paths1,0] + a,2**16) for a in Aa]).flatten()
    
    Paths=np.empty((0,0))
    
    for i in range(listSize):
        I = np.where(B==A[i,3])[0].reshape(-1,1)
        if I.size:
            I1 = np.hstack([np.mod(I,Paths1.shape[0]).reshape(-1,1),np.floor(I/Paths1.shape[0]).reshape(-1,1)]).astype(int)
            PPaths = np.hstack((Paths01267812[I1[:,0]].reshape(-1,7),Paths3459101113[I1[:,1]].reshape(-1,7),np.repeat(i,I1.shape[0]).reshape(-1,1)))
            Paths = np.vstack((Paths,PPaths)) if Paths.size else PPaths
    
    return Paths

def merge_all_paths_final(Paths01267812345910111314,A):
    
    listSize = A.shape[0]
    Paths7 = Paths01267812345910111314[:,4]
    Paths10 = Paths01267812345910111314[:,11]
    B = np.mod(A[Paths7,0] + A[Paths10,1] ,2**16)
    
    Paths=np.empty((0,0))
    
    for i in range(listSize):
        I = np.where(B==A[i,2])[0].reshape(-1,1)
        if I.size:
            PPaths = np.hstack((Paths01267812345910111314[I].reshape(-1,15),np.repeat(i,I.shape[0]).reshape(-1,1)))
            Paths = np.vstack((Paths,PPaths)) if Paths.size else PPaths
    return Paths

def pick_topKminusdelta_paths(Paths, cs_decoded_tx_message, Beta, J,K,delta):
    
    L1 = Paths.shape[0]
    LogL = np.zeros((L1,1))
    for i in range(L1):
        msg_bit=np.empty(shape=(0,0))
        path = Paths[i].reshape(1,-1)
        for j in range(path.shape[1]):
            msg_bit = np.hstack((msg_bit,j*(2**J)+cs_decoded_tx_message[path[0,j],j].reshape(1,-1))) if msg_bit.size else j*(2**J)+cs_decoded_tx_message[path[0,j],j]
            msg_bit=msg_bit.reshape(1,-1)
        LogL[i] = np.sum(np.log(Beta[msg_bit])) 
    Indices =  LogL.reshape(1,-1).argsort()[0,-(K-delta):]
    Paths = Paths[Indices,:].reshape(((K-delta),-1))
    
    return Paths

K=100 # Number of active users
B=128 # Payload size of each active user
L=16 # Number of sections/sub-blocks
n=38400 # Total number of channel uses (real d.o.f)
T=10 # Number of AMP iterations
listSize = K+10  # List size retained for each section after AMP converges
J=16  # Length of each coded sub-block
M=2**J # Length of each section
messageBlocks = np.array([1,1,0,1,1,0,1,1,0,1,1,0,0,0,0,0]).astype(int) # Indicates the indices of information blocks
# Adjacency matrix of the outer code/graph
G = np.zeros((L,L)).astype(int)
# G contains info on what parity blocks a message is attached to and what message blocks a parity is involved with
# Currently, we do not allow parity over parities. BP code needs to be modified a little to accomodate parity over parities
G[0,[2,12]]=1
G[1,[2,14]]=1
G[2,[0,1]]=1
G[3,[5,13]]=1
G[4,[5,14]]=1
G[5,[3,4]]=1
G[6,[8,12]]=1
G[7,[8,15]]=1
G[8,[6,7]]=1
G[9,[11,13]]=1
G[10,[11,14,15]]=1
G[11,[9,10]]=1
G[12,[0,6]]=1
G[13,[3,9]]=1
G[14,[1,4,10]]=1
G[15,[7,10]]=1
BPonOuterGraph = 1 # Indicates if BP is allowed on the outer code.Setting this to zero defaults back to Giuseppe's scheme that uses uninformative prior
numBPiter = 1; # Number of BP iterations on outer code. 1 seems to be good enough & AMP theory including state evolution valid only for one BP iteration
p0 = 1-(1-1/M)**K # Giuseppe's uninformative prior
delta = 0
maxSims=100 # number of simulations

def simulate(EbNodB, case):
    # EbN0 in linear scale
    EbNo = 10**(EbNodB/10)
    P = 2*B*EbNo/n
    sigma_n = 1
    
    # Power estimate
    Phat = n*P/L
    
    # variables used for measuring algorithm performance
    msgDetected = 0
    avgTime = 0
    
    # run simulation maxSims times
    for sims in range(maxSims):
        
        # Generate active users message sequences
        tx_message = np.random.randint(2, size=(K,B))

        # Outer-encode the message sequences
        encoded_tx_message_indices = Tree_encode(tx_message,K,messageBlocks,G,L,J)

        # Convert indices to sparse representation
        Beta_0 = convert_indices_to_sparse(encoded_tx_message_indices,L,J,K)
        
        # Begin case-specific code
        
        # CCS
        if case==0:
            # set up case 0
            assert n % L == 0
            numBlockRows = n//L
            
            # create Ab, Az matricies
            Ab, Az = sparc_codebook(1, M, numBlockRows, P)
            
            # obtain compressed signal to transmit through channel
            x = np.zeros((n, 1))
            for i in range(L):
                x[i*numBlockRows:(i+1)*numBlockRows] = np.sqrt(Phat)*Ab(Beta_0[i*M:(i+1)*M])
            
            # generate random channel noise and received signal
            z = np.random.randn(n, 1) * sigma_n
            y = (x + z).reshape(-1, 1)
            
            # set up AMP data structure
            Beta = np.zeros((L*M, 1))
            
            # start timer
            tic = time.time()
            
            # run CS on individual blocks using AMP.  (L independent instances of AMP)
            Beta, tau_evolution = amp(y, sigma_n, P, L, M, T, Ab, Az,p0,K,G,messageBlocks,BPonOuterGraph,numBPiter,case)
            
            # stop timer
            toc = time.time()
            
            # end case-0 specific code
        
        # CCS-Hybrid
        elif case==1:
            # set up case 1
            assert n % L == 0
            numBlockRows = n//L
            
            # create Ab, Az matricies
            Ab, Az = sparc_codebook(1, M, numBlockRows, P)
            
            # obtain compressed signal to transmit through channel
            x = np.zeros((n, 1))
            for i in range(L):
                x[i*numBlockRows:(i+1)*numBlockRows] = np.sqrt(Phat)*Ab(Beta_0[i*M:(i+1)*M])
            
            # generate random channel noise and received signal
            z = np.random.randn(n, 1) * sigma_n
            y = (x + z).reshape(-1, 1)
            
            # set up AMP data structure
            Beta = np.zeros((L*M, 1))
            
            # start timer
            tic = time.time()
            
            # run CS on individual blocks using AMP.  (L independent instances of AMP tied together by BP)
            Beta, tau_evolution = amp(y, sigma_n, P, L, M, T, Ab, Az,p0,K,G,messageBlocks,BPonOuterGraph,numBPiter,case)
            
            # stop timer
            toc = time.time()
            
            # end case-1 specific code
            
        # CCS-AMP with BP on Outer Graph
        elif case==2:
            
            # Set BPonOuterGraph = 1
            BPonOuterGraph = 1            
            
            # Generate the binned SPARC codebook
            Ab, Az = sparc_codebook(L, M, n, P)

            # Generate our transmitted signal X
            x = np.sqrt(Phat)*Ab(Beta_0)

            # Generate random channel noise and thus also received signal y
            z = np.random.randn(n, 1) * sigma_n
            y = (x + z).reshape(-1, 1)
            
            # start timer
            tic = time.time()
            
            # Run AMP decoding
            Beta, tau_evolution = amp(y, sigma_n, P, L, M, T, Ab, Az,p0,K,G,messageBlocks,BPonOuterGraph,numBPiter,case)
            
            # stop timer
            toc = time.time()
            
            # end case-2 specific code
            
        # CCS-AMP without BP on Outer Graph
        elif case==3:
            
            # Set BPonOuterGraph = 0
            BPonOuterGraph = 0
            
            # Generate the binned SPARC codebook
            Ab, Az = sparc_codebook(L, M, n, P)

            # Generate our transmitted signal X
            x = np.sqrt(Phat)*Ab(Beta_0)

            # Generate random channel noise and thus also received signal y
            z = np.random.randn(n, 1) * sigma_n
            y = (x + z).reshape(-1, 1)
            
            # start timer
            tic = time.time()
            
            # Run AMP decoding
            Beta, tau_evolution = amp(y, sigma_n, P, L, M, T, Ab, Az,p0,K,G,messageBlocks,BPonOuterGraph,numBPiter,case)
            
            # stop timer
            toc = time.time()
            
            # end case-2 specific code
        else:
            raise Exception('Invalid case')
        
        # End case-specific code
        

        # Convert decoded sparse vector into vector of indices  
        cs_decoded_tx_message = convert_sparse_to_indices(Beta,L,J,listSize)

        # Tree decoder to decode individual messages from lists output by AMP
        Paths = Tree_decoder(cs_decoded_tx_message,G,L,J,B,listSize)

        # Re-align paths to the correct order
        perm = np.argsort(np.array([0,1,2,6,7,8,12,3,4,5,9,10,11,13,14,15]))
        Paths = Paths[:,perm]

        # If tree deocder outputs more than K valid paths, retain only K of them
        if Paths.shape[0] > K:
            Paths = pick_topKminusdelta_paths(Paths, cs_decoded_tx_message, Beta, J, K,0)

        # Extract the message indices from valid paths in the tree    
        Tree_decoded_indices = extract_msg_indices(Paths,cs_decoded_tx_message, L,J)

        # Calculation of per-user prob err
        for i in range(K):
            msgDetected = msgDetected + np.equal(encoded_tx_message_indices[i,:],Tree_decoded_indices).all(axis=1).any()

        # update avgTime
        avgTime += (toc - tic)
    
    # compute error rate and average time duration
    errorRate = (K*maxSims - msgDetected)/(K*maxSims)
    avgTime /= maxSims
    
    return errorRate, avgTime

# Case 0: CCS
# Case 1: CCS-Hybrid
# Case 2: CCS-AMP, BP on Outer Graph
# Case 3: CCS-AMP, No BP on Outer Graph

numCases = 4
SNR = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4])
results = np.zeros((numCases, len(SNR)))
times = np.zeros((numCases, len(SNR)))

# Notify commencement of simulation
print('***Starting Simulations*****')

# Compare full A vs block diagonal A
for case in range(numCases):
    for idxSnr in range(len(SNR)):
        print(f'Running simulation {case*len(SNR)+idxSnr+1}/{numCases*len(SNR)}')
        results[case, idxSnr], times[case, idxSnr] = simulate(SNR[idxSnr], case)

# Notify completion
print('*****All Simulations Complete*****')

# Plot error rate results in linear scale
plt.figure(1)
plt.plot(SNR, results[0, :], 'b', label="Block Diagonal A, No BP on Outer Graph")
plt.plot(SNR, results[1, :], 'r', label="Block Diagonal A, BP on Outer Graph")
plt.plot(SNR, results[2, :], 'k', label="CCS-AMP, BP on Outer Graph")
plt.plot(SNR, results[3, :], 'g', label="CCS-AMP, No BP on Outer Graph")
plt.legend()
plt.xlabel(r'$\frac{E_b}{N_0}$')
plt.ylabel('Error Rate')
plt.title(r'Error Rates vs $\frac{E_b}{N_0}$')
plt.grid(True)

# Plot error rate results in logarithmic scale
plt.figure(2)
plt.semilogy(SNR, results[0, :], 'b', label="Block Diagonal A, No BP on Outer Graph")
plt.semilogy(SNR, results[1, :], 'r', label="Block Diagonal A, BP on Outer Graph")
plt.semilogy(SNR, results[2, :], 'k', label="CCS-AMP, BP on Outer Graph")
plt.semilogy(SNR, results[3, :], 'g', label="CCS-AMP, No BP on Outer Graph")
plt.legend()
plt.xlabel(r'$\frac{E_b}{N_0}$')
plt.ylabel('Error Rate')
plt.title(r'Error Rates vs $\frac{E_b}{N_0}$')
plt.grid(True)

# Plot time results
plt.figure(3)
plt.plot(SNR, times[0, :], 'b', label="Block Diagonal A, No BP on Outer Graph")
plt.plot(SNR, times[1, :], 'r', label="Block Diagonal A, BP on Outer Graph")
plt.plot(SNR, times[2, :], 'k', label="CCS-AMP, BP on Outer Graph")
plt.plot(SNR, times[3, :], 'g', label="CCS-AMP, No BP on Outer Graph")
plt.legend()
plt.xlabel(r'$\frac{E_b}{N_0}$')
plt.ylabel(r'Average Runtime (seconds)')
plt.title(r'Average Runtime vs $\frac{E_b}{N_0}$')
plt.grid(True)

print('Error rates: ')
print(results)
print('Times: ')
print(times)
