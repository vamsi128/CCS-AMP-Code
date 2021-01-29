"""@package ccsfg

Package @package ccsfg contains the necessary building blocks to implement a bipartite factor graph tailored
to belief propagation.
The target appication is the coded compressed sensing, which necessitates the use of a large alphabet.
Thus, the structures of @class VariableNode and @class CheckNode assume that messages are passed using
 fast Fourier transform (FFT) techniques.
"""
import numpy as np


class VariableNode:
    """
    Class @class VariableNode creates a single variable node within bipartite factor graph.
    """

    def __init__(self, varnodeid, messagelength, neighbors=None):
        """
        Initialization of variable node of type @class VariableNode.
        :param varnodeid: Unique identifier for variable node
        :param messagelength: Length of incoming and outgoing messages
        :param neighbors: Neighbors of node @var varnodeid in bipartite graph
        """

        # Unique identifier for variable node
        self.__ID = varnodeid
        # Length of messages
        self.__MessageLength = messagelength
        # Check node neighbors within bipartite graph
        # Check node identifier 0 corresponds to trivial check node associated with local observation
        self.__CheckNeighbors = [0]
        # List of messages from check node neighbors
        # Initialize messages from (trivial) check node 0 to uninformative measure (all ones)
        self.__MessagesFromChecks = [np.ones(self.__MessageLength, dtype=float)]

        # Argument @var neighbors is optional; if specified in list form, then neighbors are added
        if neighbors is not None:
            self.addneighbors(neighbors)

    def getid(self):
        return self.__ID

    def reset(self):
        """
        Reset the state of the variable node to uninformative measure (all ones)
        """
        for checkneighborid in range(len(self.__CheckNeighbors)):
            self.__MessagesFromChecks[checkneighborid] = np.ones(self.__MessageLength, dtype=float)

    def getobservation(self):
        """
        Retrieve status of local observation (checkneighborid 0)
        :return: Measure of local observation
        """
        return self.__MessagesFromChecks[0]

    def setobservation(self, measure):
        """
        Set status of local observation @var self.__CheckNeighbors[0] to @param measure.
        :param measure: Measure of local observation
        """
        self.__MessagesFromChecks[0] = measure

    def getneighbors(self):
        """
        Retrieve check node identifiers in list of neighbors.
        List item 0 corresponds to trivial check node associated with local observation.
        It should not bre returned.
        """
        return self.__CheckNeighbors[1:]

    def addneighbor(self, checkneighborid):
        """
        Add a single check neighbor @var checkneighborid to list of neighbors.
        :param checkneighborid: Unique identifier for check node to be added
        """
        if checkneighborid in self.__CheckNeighbors:
            print('Check node ID ' + str(checkneighborid) + 'is already a neighbor.')
        else:
            self.__CheckNeighbors.append(checkneighborid)
            self.__MessagesFromChecks.append(np.ones(self.__MessageLength, dtype=float))

    def addneighbors(self, checkneighborlist):
        """
        Add check node neighbors contained in @var checkneighborlist to list of neighbors.
        :param checkneighborlist: List of check node identifiers to be added as neighbors
        """
        for checkneighborid in checkneighborlist:
            self.addneighbor(checkneighborid)

    def setmessagefromcheck(self, checkneighborid, message):
        """
        Incoming message from check node neighbor @var checkneighbor to variable node @var self.__ID
        :param checkneighborid: Check node identifier of origin
        :param message: Incoming belief vector
        """
        if checkneighborid in self.__CheckNeighbors:
            neighborindex = self.__CheckNeighbors.index(checkneighborid)
            self.__MessagesFromChecks[neighborindex] = message
            # print('\t Variable ' + str(self.__ID) + ': Set message from check ' + str(checkneighbor))
            # print('\t New message: ' + str(message))
        else:
            print('Check node ID ' + str(checkneighborid) + ' is not a neighbor.')

    def getmessagetocheck(self, checkneighborid):
        """
        Outgoing message from variable node @var self.__ID to check node @var checkneighborid
        :param checkneighborid: Check node identifier of destination
        :return: Outgoing belief vector
        """
        outgoing = np.ones(self.__MessageLength, dtype=float)
        if checkneighborid in self.__CheckNeighbors:
            for other in [member for member in self.__CheckNeighbors if member is not checkneighborid]:
                otherindex = self.__CheckNeighbors.index(other)
                outgoing = np.prod((outgoing, self.__MessagesFromChecks[otherindex]), axis=0)
            # print('Variable ' + str(self.__ID) + ': Sending message to check ' + str(checkneighbor))
            # print('\t Outgoing message: ' + str(outgoing))
        else:
            print('Check node ID ' + str(checkneighborid) + ' is not a neighbor.')
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
        Retrieve distribution of beliefs associated variable node @var self.__ID.
        :return: Local belief vector
        """
        estimate = np.ones(self.__MessageLength, dtype=float)
        for checkindex in range(len(self.__CheckNeighbors)):  # self.__CheckNeighbors
            estimate = np.prod((estimate, self.__MessagesFromChecks[checkindex]), axis=0)
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
    Class @class CheckNode creates a single check node within the bipartite factor graph.
    """

    def __init__(self, checknodeid, messagelength, neighbors=None):
        """
        Initialization of check node of type @class CheckNode.
        :param checknodeid: Unique identifier for check node
        :param messagelength: Length of incoming an outgoing messages
        :param neighbors: Neighbors of node @var checknodeid in bipartite graph
        """

        # Unique identifier for check node
        self.__ID = checknodeid
        # Length of messages
        self.__MessageLength = messagelength
        # Variable node neighbors within bipartite graph
        self.__VarNeighbors = []
        # List of messages from variable node neighbors stored in FFT format
        self.__MessagesFromVarFFT = []

        # Argument @var neighbors is optional; if specified in list form, then neighbors are added
        if neighbors is not None:
            self.addneighbors(neighbors)

    def getid(self):
        return self.__ID

    def reset(self):
        """
        Reset the state of the check node to uninformative measure (FFT of all ones)
        """
        # uninformative = np.fft.fft(np.ones(self.__MessageLength, dtype=float)).real
        uninformative = np.zeros(self.__MessageLength, dtype=float)
        uninformative[0] = self.__MessageLength
        for neighbor in range(len(self.__VarNeighbors)):
            self.__MessagesFromVarFFT[neighbor] = uninformative

    def getneighbors(self):
        """
        Retrieve variable node identifiers in list of neighbors.
        """
        return self.__VarNeighbors

    def addneighbor(self, varneighborid):
        """
        Add a single variable neighbor @var varneighborid to list of neighbors.
        :param varneighborid: Unique identifier for check node to be added
        """
        if varneighborid in self.__VarNeighbors:
            print('Variable node ID ' + str(varneighborid) + 'is already a neighbor.')
        else:
            self.__VarNeighbors.append(varneighborid)
            # uninformative = np.fft.fft(np.ones(self.__MessageLength, dtype=float)).real
            uninformative = np.zeros(self.__MessageLength, dtype=float)
            uninformative[0] = self.__MessageLength
            self.__MessagesFromVarFFT.append(np.fft.fft(np.ones(self.__MessageLength, dtype=float)))

    def addneighbors(self, varneighborlist):
        """
        Add variable node neighbors contained in @var varneighborlist to list of neighbors.
        :param varneighborlist: List of variable node identifiers to be added as neighbors
        """
        for varneighborid in varneighborlist:
            self.addneighbor(varneighborid)

    def setmessagefromvar(self, varneighborid, message):
        if varneighborid in self.__VarNeighbors:
            neighborindex = self.__VarNeighbors.index(varneighborid)
            # Message from variable node @var varneighborid stored in FFT format
            self.__MessagesFromVarFFT[neighborindex] = np.fft.fft(message)
            # print('\t Check ' + str(self.__ID) + ': Set message from variable ' + str(varneighborid))
            # print('\t New message: ' + str(message))
        else:
            print('Variable node ID ' + str(varneighborid) + ' is not a neighbor.')

    def getmessagetovar(self, varneighborid):
        """
        Outgoing message from check node @var self.__ID to variable node @var varneighbor
        :param varneighborid: Variable node identifier of destination
        :return: Outgoing belief vector
        """
        outgoingFFT = np.ones(self.__MessageLength, dtype=float)
        if varneighborid in self.__VarNeighbors:
            for other in [member for member in self.__VarNeighbors if member is not varneighborid]:
                otherindex = self.__VarNeighbors.index(other)
                outgoingFFT = np.prod((outgoingFFT, self.__MessagesFromVarFFT[otherindex]), axis=0)
            # print('Check ' + str(self.__ID) + ': Sending message to variable ' + str(varneighborid))
            # print('\t Outgoing message: ' + str(outgoing))
        else:
            print('Variable node ID ' + str(varneighborid) + ' is not a neighbor.')
        outgoing = np.fft.ifft(outgoingFFT, axis=0).real
        # The outgoing message values should be indexed using the modulo operation.
        # This is implemented by retaining the value at zero and flipping the order of the remaining vector
        # That is, for n > 0, outgoing[-n % m] = np.flip(outgoing[1:])[n]
        outgoing[1:] = np.flip(outgoing[1:])  # This is implementing the required modulo operation
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
    """
    Class @class Graph creates the entire bipartite factor graph.
    """

    def __init__(self, check2varedges, varcount, infonodeindices, seclength):
        """
        Initialization of graph of type @class Graph.
        The graph is specified by passing of list of connections, one for every check node.
        The collection list for a check node contains the variable node identifiers of its neighbors.
        :param check2varedges: Connections in list of lists format
        :param varcount: Total number of variable nodes
        :param infonodeindices: List of information nodes for a systematic code
        :param seclength: Length of incoming and outgoing messages
        """
        # Number of check nodes, excluding dummy check node @var self.__CheckNodes[0]
        self.__CheckCount = len(check2varedges) - 1
        self.__CheckNodeIndices = [idx for idx in range(1, self.__CheckCount + 1)]  # IDs start at one.
        # Number of variable nodes
        self.__VarCount = varcount
        self.__VarNodeIndices = [idx for idx in range(1, self.__VarCount + 1)]  # IDs start at one.
        # Number of bits per section
        self.__SecLength = seclength
        # Length of index vector for every section
        self.__SparseSecLength = 2 ** self.__SecLength
        # Overall length of state vector
        self.__CodewordLength = self.__VarCount * self.__SparseSecLength

        # List of unique identifiers for check nodes in the graph.
        # Identifier @var checknodeid=0 is reserved for the local observation at every variable node.
        self.__CheckNodeIDs = []
        # List of check nodes in the bipartite graph, plus placeholder for local observations.
        # Check node @var self.__CheckNodes[0] is a dummy node and should not have edges.
        self.__CheckNodes = [CheckNode(0, messagelength=self.__SparseSecLength)]

        # List of unique identifiers for variable nodes in the graph.
        # The implementation assumes this set is of the form [VarCount], not including zero.
        self.__VarNodeIDs = []  # Collection of variable nodes in graph.
        # List of variable nodes in the bipartite graph.
        self.__VarNodes = []

        for varnodeid in [0] + self.__VarNodeIndices:
            if varnodeid in self.__VarNodeIDs:
                print('Variable node ID ' + str(varnodeid) + ' is already taken.')
            else:
                self.__VarNodeIDs.append(varnodeid)
                self.__VarNodes.append(VariableNode(varnodeid, messagelength=self.__SparseSecLength))

        for checknodeid in self.__CheckNodeIndices:
            if checknodeid in self.__CheckNodeIDs:
                print('Check node ID ' + str(checknodeid) + ' is already taken.')
            else:
                self.__CheckNodeIDs.append(checknodeid)
                self.__CheckNodes.append(CheckNode(checknodeid, messagelength=self.__SparseSecLength))
                self.__CheckNodes[checknodeid].addneighbors(check2varedges[checknodeid])

        for checknode in self.__CheckNodes:
            for neighbor in checknode.getneighbors():
                self.__VarNodes[neighbor].addneighbor(checknode.getid())

        self.__InfoNodeIndices = infonodeindices
        self.__InfoCount = len(self.__InfoNodeIndices)
        self.__ParityNodeIndices = [member for member in self.__VarNodeIndices if member not in self.__InfoNodeIndices]
        self.__ParityCount = len(self.__ParityNodeIndices)

    def reset(self):
        for varnode in self.__VarNodes:
            varnode.reset()
        for checknode in self.__CheckNodes:
            checknode.reset()

    def getchecklist(self):
        return self.__CheckNodeIndices

    def getcheckcount(self):
        return self.__CheckCount

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

    def getcheckneighbors(self, checknodeid):
        return self.__CheckNodes[checknodeid].getneighbors()

    def getobservation(self, varnodeid):
        if varnodeid == 0:
            print('Variable node 0 is a dummy node.')
        elif varnodeid in self.__VarNodeIndices:
            return self.__VarNodes[varnodeid].getobservation()
        else:
            print('The retrival did not succeed.')
            print('Variable Node ID: ' + str(varnodeid))

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
        for checknodeid in self.__CheckNodeIndices:
            print('Check Node ID ' + str(self.__CheckNodes[checknodeid].getid()), end=": ")
            print(self.__CheckNodes[checknodeid].getneighbors())

    def updatechecks(self, checknodelist=None):
        """
        This method updates the state of the check nodes in @var checknodelist by performing message passing.
        Every check node in @var checknodelist requests messages from its variable node neighbors.
        The received belief vectors are stored locally.
        If no list is provided, then all the check nodes in the factor graph are updated.
        :param checknodelist: List of identifiers for the check nodes to be updated
        :return: List of identifiers for the variable contacted during the update
        """
        if checknodelist is None:
            for checknode in self.__CheckNodes:
                varneighborlist = checknode.getneighbors()
                # print('Updating State of Check ' + str(checknode.getid()), end=' ')
                # print('Using Variable Neighbors ' + str(varneighborlist))
                for varnodeid in varneighborlist:
                    # print('\t Check Neighbor: ' + str(varnodeid))
                    # print('\t Others: ' + str([member for member in varneighborlist if member is not varnodeid]))
                    checknode.setmessagefromvar(varnodeid,
                                                self.__VarNodes[varnodeid].getmessagetocheck(checknode.getid()))
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
                for varnodeid in varneighborlist:
                    # print('\t Check Neighbor: ' + str(varnodeid))
                    # print('\t Others: ' + str([member for member in varneighborlist if member is not varnodeid]))
                    checknode.setmessagefromvar(varnodeid,
                                                self.__VarNodes[varnodeid].getmessagetocheck(checknode.getid()))
            return list(varneighborsaggregate)

    def updatevars(self, varnodelist=None):
        """
        This method updates the state of the variable nodes in @var varnodelist by performing message passing.
        Every variable node in @var varnodelist requests messages from its variable node neighbors.
        The received belief vectors are stored locally.
        If no list is provided, then all the variable nodes in the factor graph are updated.
        :param varnodelist: List of identifiers for the variable nodes to be updated
        :return: List of identifiers for the check contacted during the update
        """
        if varnodelist is None:
            for varnode in self.__VarNodes:
                checkneighborlist = varnode.getneighbors()
                # print('Updating State of Variable ' + str(varnode.getid()), end=' ')
                # print('Using Check Neighbors ' + str(checkneighborlist))
                for checknodeid in checkneighborlist:
                    # print('\t Variable Neighbor: ' + str(neighbor))
                    # print('\t Others: ' + str([member for member in checkneighborlist if member is not neighbor]))
                    varnode.setmessagefromcheck(checknodeid,
                                                self.__CheckNodes[checknodeid].getmessagetovar(varnode.getid()))
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
                for checknodeid in checkneighborlist:
                    # print('\t Variable Neighbor: ' + str(neighbor))
                    # print('\t Others: ' + str([member for member in checkneighborlist if member is not neighbor]))
                    varnode.setmessagefromcheck(checknodeid,
                                                self.__CheckNodes[checknodeid].getmessagetovar(varnode.getid()))
            return list(checkneighborsaggregate)

    def getestimate(self, varnodeid):
        """
        This method returns the belief vector associated with variable node @var varnodeid.
        :param varnodeid: Identifier of the variable node to be queried
        :return: Belief vector from variable node @var varnodeid
        """
        varnode = self.__VarNodes[varnodeid]
        return varnode.getestimate()

    def getestimates(self):
        """
        This method returns belief vectors for all the variable nodes in the bipartite graph.
        :return: Array of belief vectors from all variable nodes
        """
        estimates = np.empty((self.__VarCount, self.__SparseSecLength), dtype=float)
        for varnode in self.__VarNodes[1:]:
            estimates[varnode.getid() - 1] = varnode.getestimate()
        return estimates

    def getextrinsicestimate(self, varnodeid):
        """
        This method returns the belief vector associated with variable node @var varnodeid,
        based only on extrinsic information.
        It does not incorporate information from the local observation @var checknode zero.
        :param varnodeid: Identifier of the variable node to be queried
        :return:
        """
        return self.__VarNodes[varnodeid].getmessagetocheck(0)

    def getcodeword(self):
        """

        :return:
        """
        codeword = np.empty((self.__VarCount, self.__SparseSecLength), dtype=int)
        for varnode in self.__VarNodes[1:]:  # self.__VarNodes[0] is a dummy codeword
            block = np.zeros(self.__SparseSecLength, dtype=int)
            if np.max(varnode.getestimate()) > 0:
                block[np.argmax(varnode.getestimate())] = 1
            codeword[varnode.getid() - 1] = block
        return codeword

    def encodemessage(self, bits):
        if len(bits) == (self.__InfoCount * self.__SecLength):
            bitsections = np.resize(bits, [self.__InfoCount, self.__SecLength])
            self.reset()  # Reinitialize factor graph before encoding
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
            for idx in range(3):  # Need to change this
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

    def testvalid(self, codeword):  # ISSUE IN USING THIS FOR NOISY CODEWORDS, INPUT SHOULD BE MEASURE
        self.reset()
        if len(codeword) == (self.__CodewordLength):
            bitsections = np.resize(codeword, [self.__VarCount, self.__SparseSecLength])
            for varnodeid in self.__VarNodeIndices:
                self.setobservation(varnodeid, bitsections[varnodeid - 1])
                # print('Variable node ' + str(varnodeid), end=' ')
                # print(' -- Observation changed to: ' + str(np.argmax(self.getobservation(varnodeid))))

            for idx in range(16):
                # print(self.getestimates())
                self.updatechecks()
                self.updatevars()
        else:
            print('Issue')
        return self.getcodeword()
