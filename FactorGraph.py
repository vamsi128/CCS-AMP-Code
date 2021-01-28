__author__ = 'JF Chamberland'

import numpy as np

VarNodeIDs = []  # Collection of variable nodes in graph.
CheckNodeIDs = []  # Collection of factor nodes in graph


class VariableNode:
    """
    This class creates a variable node for a bipartite factor graph.
    """

    def __init__(self, varnodeid, messagelength, neighbors=None):

        self.__ID = varnodeid
        self.__MessageLength = messagelength
        self.__CheckNeighbors = [0]   # Neighbor 0 is trivial check node associated with local observation.
        self.__MessagesFromChecks = [np.ones(self.__MessageLength, dtype=float)]   # Incoming messages from check neighbors.

        if varnodeid in VarNodeIDs:
            print('Variable node ID ' + str(varnodeid) + ' is already taken.')
        else:
            VarNodeIDs.append(self.__ID)

        if neighbors is not None:
            self.addneighbors(neighbors)

    def getid(self):
        return self.__ID

    def reset(self):
        """
        Reset the state of the variable node.
        :return:
        """
        for neighbor in range(len(self.__CheckNeighbors)):
            self.__MessagesFromChecks[neighbor] = np.ones(self.__MessageLength, dtype=float)

    def setobservation(self, measure):
        """
        Sets local observation of trivial check node.
        :param measure: Measure of local observation
        """
        self.__MessagesFromChecks[0] = measure

    def getobservation(self):
        return self.__MessagesFromChecks[0]

    def addneighbor(self, neighbor):
        if neighbor in self.__CheckNeighbors:
            print('Check node ID ' + str(neighbor) + 'is already a neighbor.')
        else:
            self.__CheckNeighbors.append(neighbor)
            self.__MessagesFromChecks.append(np.ones(self.__MessageLength, dtype=float))

    def addneighbors(self, neighbors):
        for neighbor in neighbors:
            self.addneighbor(neighbor)

    def getneighbors(self):
        return self.__CheckNeighbors[1:]  # self.__NeighborList[0] is the trivial check node of local observation.

    def setmessagefromcheck(self, checkneighbor, message):
        """
        Incoming message from check node enighbor to variable node self.__ID
        :param checkneighbor: Origin check node
        :param message: Incoming belief vectr
        """
        if checkneighbor in self.__CheckNeighbors:
            neighborindex = self.__CheckNeighbors.index(checkneighbor)
            self.__MessagesFromChecks[neighborindex] = message
            # print('\t Variable ' + str(self.__ID) + ': Set message from check ' + str(checkneighbor))
            # print('\t New message: ' + str(message))
        else:
            print('Check node ID ' + str(checkneighbor) + ' is not a neighbor.')

    def getmessagetocheck(self, checkneighbor):
        """
        Outgoing message from variable node self.__ID to check node checkneighbor
        :param checkneighbor: Destination check node
        :return: Outgoing belief vector
        """
        outgoing = np.ones(self.__MessageLength, dtype=float)
        if checkneighbor in self.__CheckNeighbors:
            for other in [member for member in self.__CheckNeighbors if member is not checkneighbor]:
                otherindex = self.__CheckNeighbors.index(other)
                outgoing = np.prod((outgoing,self.__MessagesFromChecks[otherindex]), axis=0)
        else:
            print('Check node ID ' + str(checkneighbor) + ' is not a neighbor.')
        # print('Variable ' + str(self.__ID) + ': Sending message to check ' + str(checkneighbor))
        # print('\t Outgoing message: ' + str(outgoing))
        try:
            normalization = (1 / np.linalg.norm(outgoing, ord=1))
        except ZeroDivisionError as e:
            print(e)
            normalization = 0
        if np.isfinite(normalization):
            return normalization * outgoing
        else:
            return np.zeros(self.__MessageLength, dtype=float)

    def getestimate(self):
        """
        Outgoing message from variable node self.__ID to check node checkneighbor
        :param checkneighbor: Destination check node
        :return: Outgoing belief vector
        """
        estimate = np.ones(self.__MessageLength, dtype=float)
        for checkindex in range(len(self.__CheckNeighbors)):
            estimate = np.prod((estimate,self.__MessagesFromChecks[checkindex]), axis=0)
        try:
            normalization = (1 / np.linalg.norm(estimate, ord=1))
        except ZeroDivisionError as e:
            print(e)
            normalization = 0
        if np.isfinite(normalization):
            return normalization * estimate
        else:
            return np.zeros(self.__MessageLength, dtype=float)


class CheckNode:
    """
    This class creates a factor node for a bipartite factor graph.
    """

    def __init__(self, checknodeid, messagelength, neighbors=None):

        self.__ID = checknodeid
        self.__MessageLength = messagelength
        self.__VarNeighbors = []
        self.__MessagesFromVarFFT = []   # Stored in FFT format.

        if checknodeid in CheckNodeIDs:
            print('Check node ID ' + str(checknodeid) + ' is already taken.')
        else:
            CheckNodeIDs.append(self.__ID)

        if neighbors is not None:
            self.addneighbors(neighbors)

    def getid(self):
        return self.__ID

    def reset(self):
        """
        Reset the state of the check node.
        :return:
        """
        uninformative = np.fft.fft(np.ones(self.__MessageLength, dtype=float))
        for neighbor in range(len(self.__VarNeighbors)):
            self.__MessagesFromVarFFT[neighbor] = uninformative

    def getneighbors(self):
        return self.__VarNeighbors

    def addneighbor(self, neighbor):
        if neighbor in self.__VarNeighbors:
            print('Variable node ID ' + str(neighbor) + 'is already a neighbor.')
        else:
            self.__VarNeighbors.append(neighbor)
            self.__MessagesFromVarFFT.append(np.fft.fft(np.ones(self.__MessageLength, dtype=float)))   # Should be updated before use

    def addneighbors(self, neighbors):
        for neighbor in neighbors:
            self.addneighbor(neighbor)

    def setmessagefromvar(self, varneighbor, message):
        if varneighbor in self.__VarNeighbors:
            neighborindex = self.__VarNeighbors.index(varneighbor)
            self.__MessagesFromVarFFT[neighborindex] = np.fft.fft(message)   # Stored in FFT format.
            # print('\t Check ' + str(self.__ID) + ': Set message from variable ' + str(varneighbor))
            # print('\t New message: ' + str(message))
        else:
            print('Variable node ID ' + str(varneighbor) + ' is not a neighbor.')

    def getmessagetovar(self, varneighbor):
        """
        Outgoing message from check node self.__ID to variable node varneighbor
        :param varneighbor: Destination variable node
        :return: Outgoing belief vector
        """
        outgoingFFT = np.ones(self.__MessageLength, dtype=float)
        if varneighbor in self.__VarNeighbors:
            for other in [member for member in self.__VarNeighbors if member is not varneighbor]:
                otherindex = self.__VarNeighbors.index(other)
                outgoingFFT = np.prod((outgoingFFT,self.__MessagesFromVarFFT[otherindex]), axis=0)
        else:
            print('Variable node ID ' + str(varneighbor) + ' is not a neighbor.')
        outgoing = np.fft.ifft(outgoingFFT, axis=0).real
        outgoing[1:] = np.flip(outgoing[1:])    # This is implementing the required modulo operation
        # print('Check ' + str(self.__ID) + ': Sending message to variable ' + str(varneighbor))
        # print('\t Outgoing message: ' + str(outgoing))
        try:
            normalization = (1 / np.linalg.norm(outgoing, ord=1))
        except ZeroDivisionError as e:
            print(e)
            normalization = 0
        if np.isfinite(normalization):
            return normalization * outgoing
        else:
            return np.zeros(self.__MessageLength, dtype=float)


class Graph:

    def __init__(self, check2varedges, varcount, infonodeindices, seclength):
        self.__CheckCount = len(check2varedges) - 1
        self.__CheckNodeIndices = [id for id in range(1,self.__CheckCount+1)] # IDs start at one.
        self.__VarCount = varcount
        self.__VarNodeIndices = [id for id in range(1,self.__VarCount+1)] # IDs start at one.
        self.__InfoNodeIndices = infonodeindices
        self.__InfoCount = len(self.__InfoNodeIndices)
        self.__ParityNodeIndices = [member for member in self.__VarNodeIndices if member not in self.__InfoNodeIndices]
        self.__ParityCount = len(self.__ParityNodeIndices)
        self.__SecLength = seclength
        self.__SparseSecLength = 2 ** self.__SecLength
        self.__CodewordLength = self.__VarCount * self.__SparseSecLength

        self.__VarNodes = [VariableNode(l, messagelength=self.__SparseSecLength) for l in [0] + self.__VarNodeIndices]
        self.__CheckNodes = [CheckNode(0, messagelength=self.__SparseSecLength)] # Node 0 is a dummy node.
        for id in self.__CheckNodeIndices:
            self.__CheckNodes.append(CheckNode(id, messagelength=self.__SparseSecLength))
            self.__CheckNodes[id].addneighbors(check2varedges[id])

        for checknode in self.__CheckNodes:
            for neighbor in checknode.getneighbors():
                self.__VarNodes[neighbor].addneighbor(checknode.getid())

    def reset(self):
        for varnode in self.__VarNodes:
            varnode.reset()
        for checknode in self.__CheckNodes:
            checknode.reset()

    def getvarlist(self):
        return self.__VarNodeIndices

    def getvarcount(self):
        return self.__VarCount

    def getinfolist(self):
        return self.__InfoNodeIndices

    def getinfocount(self):
        return self.__InfoCount

    def getparitylist(self):
        return self.__ParityNodeIndices

    def getseclength(self):
        return self.__SecLength

    def getsparseseclength(self):
        return self.__SparseSecLength

    def getobservation(self, varnodeid):
        if varnodeid == 0:
            print('Variable node 0 is a dummy node.')
        elif varnodeid in self.__VarNodeIndices:
            return self.__VarNodes[varnodeid].getobservation()
        else:
            print('The retrival did not succeed.')
            print('Variable Node ID: ' + str(variablenode))

    def setobservation(self, varnodeid, measure):
        if varnodeid == 0:
            print('Variable node 0 is a dummy node.')
        elif (len(measure) == self.__SparseSecLength) and (varnodeid in self.__VarNodeIndices):
            self.__VarNodes[varnodeid].setobservation(measure)
        else:
            print('The assignment did not succeed.')
            print('Variable Node ID: ' + str(varnodeid))
            print('Variable Node Indices: ' + str(self.__VarNodeIndices))
            print('Length Measure: ' + str(len(measure)))
            print('Length Sparse Section: ' + str(self.__SparseSecLength))

    def printgraph(self):
        for varnode in self.__VarNodeIndices:
            print('Var Node ID ' + str(self.__VarNodes[varnode].getid()), end=": ")
            print(self.__VarNodes[varnode].getneighbors())
        for checknode in self.__CheckNodeIndices:
            print('Check Node ID ' + str(self.__CheckNodes[checknode].getid()), end=": ")
            print(self.__CheckNodes[checknode].getneighbors())

    def updatechecks(self,checknodelist=None):
        if checknodelist is None:
            for checknode in self.__CheckNodes:
                varneighborlist = checknode.getneighbors()
                # print('Updating State of Check ' + str(checknode.getid()), end=' ')
                # print('Using Variable Neighbors ' + str(varneighborlist))
                for varneighbor in varneighborlist:
                    # print('\t Check Neighbor: ' + str(varneighbor))
                    # print('\t Others: ' + str([member for member in varneighborlist if member is not varneighbor]))
                    checknode.setmessagefromvar(varneighbor, self.__VarNodes[varneighbor].getmessagetocheck(checknode.getid()))
            return
        else:
            varneighborsaggregate = set()
            for checknodeid in checknodelist:
                try:
                    checknode = self.__CheckNodes[checknodeid]
                except IndexError as e:
                    print('Check node ID ' + str(checknodeid) + ' is not in ' + str(checknodelist))
                    print('IndexError: ' + str(e))
                    break
                varneighborlist = checknode.getneighbors()
                varneighborsaggregate.update(varneighborlist)
                # print('Updating State of Check ' + str(checknode.getid()), end=' ')
                # print('Using Variable Neighbors ' + str(varneighborlist))
                for varneighbor in varneighborlist:
                    # print('\t Check Neighbor: ' + str(varneighbor))
                    # print('\t Others: ' + str([member for member in varneighborlist if member is not varneighbor]))
                    checknode.setmessagefromvar(varneighbor, self.__VarNodes[varneighbor].getmessagetocheck(checknode.getid()))
            return list(varneighborsaggregate)

    def updatevars(self,varnodelist=None):
        if varnodelist is None:
            for varnode in self.__VarNodes:
                checkneighborlist = varnode.getneighbors()
                # print('Updating State of Variable ' + str(varnode.getid()), end=' ')
                # print('Using Check Neighbors ' + str(checkneighborlist))
                for checkneighbor in checkneighborlist:
                    # print('\t Variable Neighbor: ' + str(neighbor))
                    # print('\t Others: ' + str([member for member in checkneighborlist if member is not neighbor]))
                    varnode.setmessagefromcheck(checkneighbor, self.__CheckNodes[checkneighbor].getmessagetovar(varnode.getid()))
            return
        else:
            checkneighborsaggregate = set()
            for varnodeid in varnodelist:
                try:
                    varnode = self.__VarNodes[varnodeid]
                except IndexError as e:
                    print('Check node ID ' + str(varnodeid) + ' is not in ' + str(varnodelist))
                    print('IndexError: ' + str(e))
                    break
                checkneighborlist = varnode.getneighbors()
                checkneighborsaggregate.update(checkneighborlist)
                # print('Updating State of Variable ' + str(varnode.getid()), end=' ')
                # print('Using Check Neighbors ' + str(checkneighborlist))
                for checkneighbor in checkneighborlist:
                    # print('\t Variable Neighbor: ' + str(neighbor))
                    # print('\t Others: ' + str([member for member in checkneighborlist if member is not neighbor]))
                    varnode.setmessagefromcheck(checkneighbor, self.__CheckNodes[checkneighbor].getmessagetovar(varnode.getid()))
            return list(checkneighborsaggregate)

    def getcodeword(self):
        codeword = np.empty((self.__VarCount, self.__SparseSecLength), dtype=int)
        for varnode in self.__VarNodes[1:]: # self.__VarNodes[0] is a dummy codeword
            block = np.zeros(self.__SparseSecLength, dtype=int)
            if np.max(varnode.getestimate()) > 0:
                block[np.argmax(varnode.getestimate())] = 1
            codeword[varnode.getid()-1] = block
        return codeword

    def getestimate(self, varnodeid):
        varnode = self.__VarNodes[varnodeid]
        return varnode.getestimate()

    def getestimates(self):
        estimates = np.empty((self.__VarCount, self.__SparseSecLength), dtype=float)
        for varnode in self.__VarNodes[1:]:
            estimates[varnode.getid()-1] = varnode.getestimate()
        return estimates.reshape(1, -1)

    def getextrinsicestimate(self,varnodeid):
        return self.__VarNodes[varnodeid].getmessagetocheck(0)

    def encodemessage(self, bits):
        if len(bits) == (self.__InfoCount * self.__SecLength):
            bitsections = np.resize(bits, [self.__InfoCount, self.__SecLength])
            self.reset()    # Reinitialize factor graph before encoding
            for varnodeindex in range(self.__InfoCount):
                varnodeid = self.__InfoNodeIndices[varnodeindex]
                fragment = np.inner(bitsections[varnodeindex], 2 ** np.arange(self.__SecLength))
                # fragment = np.inner(bitsections[varnodeindex], 2 ** np.arange(self.__SecLength)[::-1]) # CHECK
                sparsefragment = np.zeros(self.__SparseSecLength, dtype=int)
                sparsefragment[fragment] = 1
                self.setobservation(varnodeid, sparsefragment)
                # print('Variable node ' + str(varnodeid), end=' ')
                # print(' -- Observation changed to: ' + str(np.argmax(self.getobservation(varnodeid))))

            varnodesvisited = set(self.__VarNodeIndices)
            checknodesvisited = set([])
            varnodes2visitnext = set(varnodesvisited)
            checknodes2visitnext = set([])
            for iter in range(3): # Need to change this
                self.updatechecks()  # Update Check first
                self.updatevars()
                # checknodes2visitnext = set(self.updatevars(varnodes2visitnext)) - checknodesvisited
                # checknodesvisited = checknodesvisited - checknodes2visitnext
                # # print(self.updatechecks(checknodes2visitnext))
                # varnodes2visitnext = set(self.updatechecks(checknodes2visitnext))- varnodesvisited
                # # print(self.updatevars(varnodes2visitnext))
                # varnodesvisited = varnodesvisited | varnodes2visitnext

            return self.getcodeword()
        else:
            print('Length of input array is not ' + str(self.getinfocount() * self.getseclength()))

    def encodemessages(self, infoarray):
        codewords = []
        for messageindex in range(len(infoarray)):
            codewords.append(self.encodemessage(infoarray[messageindex]).flatten().astype(int))
        return np.asarray(codewords)

    def encodesignal(self, infoarray):
        signal = [np.zeros(self.__SparseSecLength, dtype=float) for l in range(self.__VarCount)]
        for messageindex in range(len(infoarray)):
            signal = signal + self.encodemessage(infoarray[messageindex]).astype(int)
        return signal



    def testvalid(self, codeword): # ISSUE IN USING THIS FOR NOISY CODEWORDS, INPUT SHOULD BE MEASURE
        self.reset()
        if len(codeword) == (self.__CodewordLength):
            bitsections = np.resize(codeword, [self.__VarCount, self.__SparseSecLength])
            for varnodeid in self.__VarNodeIndices:
                self.setobservation(varnodeid, bitsections[varnodeid-1])
                # print('Variable node ' + str(varnodeid), end=' ')
                # print(' -- Observation changed to: ' + str(np.argmax(self.getobservation(varnodeid))))

            for iter in range(16):
                # print(self.getestimates())
                self.updatechecks()
                self.updatevars()
        else:
            print('Issue')
        return self.getcodeword()
