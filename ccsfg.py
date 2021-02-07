"""@package ccsfg

Package @package ccsfg contains the necessary building blocks to implement a bipartite factor graph tailored
to belief propagation.
The target appication is the coded compressed sensing, which necessitates the use of a large alphabet.
Thus, the structures of @class VariableNode and @class CheckNode assume that messages are passed using
 fast Fourier transform (FFT) techniques.
"""
import numpy as np


class GenericNode:
    """
    Class @class GenericNode creates a single generic node within graph.
    """

    def __init__(self, nodeid, neighbors=None):
        """
        Initialize node of type @class GenericNode.
        :param nodeid: Identifier corresponding to self
        :param neighbors: List of identifiers corresponding to neighbors of self
        """

        # Identifier of self
        self.__ID = nodeid
        # List of identifiers corresponding to neighbors within graph
        self.__Neighbors = []
        # Dictionary of messages from neighbors acces with their identifiers
        # Some neighbors may not have set messages
        # Therefore self.__Neighbors may not match self.__MessagesFromNeighbors.keys()
        self.__MessagesFromNeighbors = dict()

        # Argument @var neighbors (optional) if specified in list form, then neighbors are added
        if neighbors is not None:
            self.addneighbors(neighbors)

    def getid(self):
        return self.__ID

    def getneighbors(self):
        """
        Retrieve node identifiers contained in list of neighbors.
        """
        return self.__Neighbors

    def addneighbor(self, neighborid, message=None):
        """
        Add neighbor @var neighborid to list of neighbors.
        Add message @var message (optional) to dictionary of messages from neighbors.
        :param neighborid: Identifier of neighbor to be added
        :param message: Message associated with @var neighborid
        """
        if neighborid in self.__Neighbors:
            print('Node ID ' + str(neighborid) + 'is already a neighbor.')
        else:
            if message is None:
                self.__Neighbors.append(neighborid)
            else:
                self.__MessagesFromNeighbors.update({neighborid: message})
                self.__Neighbors.append(neighborid)

    def addneighbors(self, neighborlist):
        """
        Add neighbors whose identifiers are contained in @var neighborlist to list of neighbors.
        :param neighborlist: List of node identifiers to be added as neighbors
        """
        for neighborid in neighborlist:
            self.addneighbor(neighborid)

    def getstate(self, neighborid):
        """
        Output message corresponding to @var nodeid.
        :param neighborid:
        :return:
        """
        if neighborid in self.__MessagesFromNeighbors.keys():
            return self.__MessagesFromNeighbors[neighborid]
        else:
            return None

    def getstates(self):
        """
        Output @var self.__MessagesFromNeighbors in dictionary format.
        :return: Dictionary of messages from neighbors
        """
        return self.__MessagesFromNeighbors

    def setstate(self, neighborid, message):
        """
        set message for neighbor with identifier @var neighborid.
        :param neighborid: Identifier of origin
        :param message: Message corresponding to identifier @var neighborid
        """
        if neighborid in self.__Neighbors:
            self.__MessagesFromNeighbors[neighborid] = message
        else:
            print('Check node ID ' + str(neighborid) + ' is not a neighbor.')


class VariableNodeFFT(GenericNode):
    """
    Class @class VariableNode creates a single variable node within bipartite factor graph.
    """

    def __init__(self, varnodeid, messagelength, neighbors=None):
        """
        Initialize variable node of type @class VariableNode.
        :param varnodeid: Unique identifier for variable node
        :param messagelength: Length of incoming and outgoing messages
        :param neighbors: Neighbors of node @var varnodeid in bipartite graph
        """

        super().__init__(varnodeid, neighbors)
        # Unique identifier for variable node
        self.__ID = varnodeid
        # Length of messages
        self.__MessageLength = messagelength

        # Check node identifier 0 corresponds to trivial check node associated with local observation
        # Initialize messages from (trivial) check node 0 to uninformative measure (all ones)
        self.addneighbor(0, message=np.ones(self.__MessageLength, dtype=float))

    def reset(self):
        """
        Reset every state of variable node to uninformative measures (all ones).
        This method employs super().getneighbors() to properly reset message for
        (trivial) check node zero to uninformative measure.
        """
        for neighborid in super().getneighbors():
            self.setstate(neighborid, np.ones(self.__MessageLength, dtype=float))
        # self.setobservation(self, np.ones(self.__MessageLength, dtype=float))

    def getneighbors(self):
        """
        Retrieve node identifiers contained in list of neighbors.
        """
        return [neighbor for neighbor in super().getneighbors() if neighbor != 0]

    def getobservation(self):
        """
        Retrieve status of local observation (checkneighborid 0)
        :return: Measure of local observation
        """
        return self.getstate(0)

    def setobservation(self, measure):
        """
        Set status of local observation @var self.__CheckNeighbors[0] to @param measure.
        :param measure: Measure of local observation
        """
        self.setstate(0, measure)

    def setmessagefromcheck(self, checkneighborid, message):
        """
        Incoming message from check node neighbor @var checkneighbor to variable node self.
        :param checkneighborid: Check node identifier of origin
        :param message: Incoming belief vector
        """
        self.setstate(checkneighborid, message)

    def getmessagetocheck(self, checkneighborid=None):
        """
        Outgoing message from variable node self to check node @var checkneighborid
        Exclude message corresponding to @var checkneighborid (optional).
        If no destination is specified, return product of all measures.
        :param checkneighborid: Check node identifier of destination
        :return: Outgoing belief vector
        """
        dictionary = self.getstates()
        if checkneighborid is None:
            states = list(dictionary.values())
        elif checkneighborid in dictionary:
            states = [dictionary[key] for key in dictionary if key is not checkneighborid]
        else:
            print('Desination check node ID ' + str(checkneighborid) + ' is not a neighbor.')
            return None

        if np.isscalar(states):
            return states
        else:
            states = np.array(states)
            if states.ndim == 1:
                return states
            elif states.ndim == 2:
                try:
                    return np.prod(states, axis=0)
                except ValueError as e:
                    print(e)
            else:
                raise RuntimeError('Dimenstion: states.ndim = ' + str(np.array(states).ndim) + ' is not allowed.')

    def getestimate(self):
        """
        Retrieve distribution of beliefs associated with self
        :return: Local belief distribution
        """
        measure = self.getmessagetocheck()
        if measure is None:
            return measure
        elif np.isscalar(measure):
            return measure
        else:
            try:
                return measure / np.linalg.norm(measure, ord=1)
            except ZeroDivisionError:
                return measure


class CheckNodeFFT(GenericNode):
    """
    Class @class CheckNode creates a single check node within bipartite factor graph.
    """

    def __init__(self, checknodeid, messagelength, neighbors=None):
        """
        Initialize check node of type @class CheckNode.
        :param checknodeid: Unique identifier for check node
        :param messagelength: Length of incoming and outgoing messages
        :param neighbors: Neighbors of node @var checknodeid in bipartite graph
        """

        super().__init__(checknodeid, neighbors)
        # Unique identifier for check node
        self.__ID = checknodeid
        # Length of messages
        self.__MessageLength = messagelength

    def reset(self):
        """
        Reset every states check node to uninformative measures (FFT of all ones)
        """
        # uninformative = np.fft.rfft(np.ones(self.__MessageLength, dtype=float))
        uninformative = np.zeros(self.__MessageLength, dtype=float)
        uninformative[0] = self.__MessageLength
        for neighborid in self.getneighbors():
            self.setstate(neighborid, uninformative)

    def setmessagefromvar(self, varneighborid, message):
        """
        Incoming message from variable node neighbor @var vaneighborid to check node self.
        :param varneighborid: Variable node identifier of origin
        :param message: Incoming belief vector
        """
        self.setstate(varneighborid, np.fft.rfft(message))

    def getmessagetovar(self, varneighborid):
        """
        Outgoing message from check node self to variable node @var varneighbor
        :param varneighborid: Variable node identifier of destination
        :return: Outgoing belief vector
        """
        dictionary = self.getstates()
        if varneighborid is None:
            states = list(dictionary.values())
        elif varneighborid in dictionary:
            states = [dictionary[key] for key in dictionary if key is not varneighborid]
        else:
            print('Destination variable node ID ' + str(varneighborid) + ' is not a neighbor.')
            return None
        if np.isscalar(states):
            return states
        else:
            states = np.array(states)
            if states.ndim == 1:
                outgoing_fft = states
            elif states.ndim == 2:
                try:
                    outgoing_fft = np.prod(states, axis=0)
                except ValueError as e:
                    print(e)
                    return None
            else:
                raise RuntimeError('states.ndim = ' + str(np.array(states).ndim) + ' is not allowed.')
            outgoing = np.fft.irfft(outgoing_fft, axis=0)
        # The outgoing message values should be indexed using the modulo operation.
        # This is implemented by retaining the value at zero and flipping the order of the remaining vector
        # That is, for n > 0, outgoing[-n % m] = np.flip(outgoing[1:])[n]
        outgoing[1:] = np.flip(outgoing[1:])  # This is implementing the required modulo operation
        return outgoing


class Graph:
    """
    Class @class Graph creates bipartite factor graph for belief propagation.
    """

    def __init__(self, check2varedges, seclength):
        """
        Initialize bipartite graph of type @class Graph.
        Graph is specified by passing list of connections, one for every check node.
        The list for every check node contains the variable node identifiers of its neighbors.
        :param check2varedges: Edges from check nodes to variable nodes in list of lists format
        :param seclength: Length of incoming and outgoing messages
        """
        # Number of bits per section.
        self.__SecLength = seclength
        # Length of index vector for every section.
        self.__SparseSecLength = 2 ** self.__SecLength

        # List of unique identifiers for check nodes in bipartite graph.
        # Identifier @var checknodeid=0 is reserved for the local observation at every variable node.
        self.__CheckNodeIDs = set()
        # Dictionary of identifiers and nodes for check nodes in bipartite graph.
        self.__CheckNodes = dict()

        # List of unique identifiers for variable nodes in bipartite graph.
        self.__VarNodeIDs = set()
        # Dictionary of identifiers and nodes for variable nodes in bipartite graph.
        self.__VarNodes = dict()

        # Check identifier @var checknodeid=0 is reserved for the local observation at every variable node.
        for idx in range(len(check2varedges)):
            # Create check node identifier, starting at @var checknodeid = 1.
            checknodeid = idx + 1
            self.__CheckNodeIDs.add(checknodeid)
            # Create check nodes and add them to dictionary @var self.__CheckNodes.
            self.__CheckNodes.update({checknodeid: CheckNodeFFT(checknodeid, messagelength=self.__SparseSecLength)})
            # Add edges from check nodes to variable nodes.
            self.__CheckNodes[checknodeid].addneighbors(check2varedges[idx])
            # Create set of all variable node identifiers.
            self.__VarNodeIDs.update(check2varedges[idx])

        for varnodeid in self.__VarNodeIDs:
            # Create variable nodes and add them to dictionary @var self.__VariableNodes.
            self.__VarNodes[varnodeid] = VariableNodeFFT(varnodeid, messagelength=self.__SparseSecLength)

        for checknode in self.__CheckNodes.values():
            for neighbor in checknode.getneighbors():
                # Add edges from variable nodes to check nodes.
                self.__VarNodes[neighbor].addneighbor(checknode.getid())

    def reset(self):
        # Reset states at variable nodes to uniform measures.
        for varnode in self.__VarNodes.values():
            varnode.reset()
        # Reset states at check nodes to uninformative measures.
        for checknode in self.__CheckNodes.values():
            checknode.reset()

    def getchecklist(self):
        return list(self.__CheckNodes.keys())

    def getcheckcount(self):
        return len(self.__CheckNodes)

    def getvarlist(self):
        return list(self.__VarNodes.keys())

    def getvarcount(self):
        return len(self.__VarNodes)

    def getseclength(self):
        return self.__SecLength

    def getsparseseclength(self):
        return self.__SparseSecLength

    def getcheckneighbors(self, checknodeid):
        return self.__CheckNodes[checknodeid].getneighbors()

    def getobservation(self, varnodeid):
        if varnodeid in self.getvarlist():
            return self.__VarNodes[varnodeid].getobservation()
        else:
            print('The retrival did not succeed.')
            print('Variable Node ID: ' + str(varnodeid))

    def setobservation(self, varnodeid, measure):
        if (len(measure) == self.getsparseseclength()) and (varnodeid in self.getvarlist()):
            self.__VarNodes[varnodeid].setobservation(measure)
        else:
            print('The assignment did not succeed.')
            print('Variable Node ID: ' + str(varnodeid))
            print('Variable Node Indices: ' + str(self.getvarlist()))
            print('Length Measure: ' + str(len(measure)))
            print('Length Sparse Section: ' + str(self.__SparseSecLength))

    def printgraph(self):
        for varnodeid in self.getvarlist():
            print('Var Node ID ' + str(varnodeid), end=": ")
            print(self.__VarNodes[varnodeid].getneighbors())
        for checknodeid in self.getchecklist():
            print('Check Node ID ' + str(checknodeid), end=": ")
            print(self.__CheckNodes[checknodeid].getneighbors())

    def printgraphcontent(self):
        for varnodeid in self.getvarlist():
            print('Var Node ID ' + str(varnodeid), end=": ")
            print(self.__VarNodes[varnodeid].getstates())
        for checknodeid in self.getchecklist():
            print('Check Node ID ' + str(checknodeid), end=": ")
            print(self.__CheckNodes[checknodeid].getstates())

    def updatechecks(self, checknodelist=None):
        """
        This method updates states of check nodes in @var checknodelist by performing message passing.
        Every check node in @var checknodelist requests messages from its variable node neighbors.
        The received belief vectors are stored locally.
        If no list is provided, then all check nodes in the factor graph are updated.
        :param checknodelist: List of identifiers for check nodes to be updated
        :return: List of identifiers for variable node contacted during update
        """
        if checknodelist is None:
            for checknode in self.__CheckNodes.values():
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
        This method updates states of variable nodes in @var varnodelist by performing message passing.
        Every variable node in @var varnodelist requests messages from its check node neighbors.
        The received belief vectors are stored locally.
        If no list is provided, then all variable nodes in factor graph are updated.
        :param varnodelist: List of identifiers for variable nodes to be updated
        :return: List of identifiers for check node contacted during update
        """
        if varnodelist is None:
            for varnode in self.__VarNodes.values():
                checkneighborlist = varnode.getneighbors()
                # print('Updating State of Variable ' + str(varnode.getid()), end=' ')
                # print('Using Check Neighbors ' + str(checkneighborlist))
                for checknodeid in checkneighborlist:
                    # print('\t Variable Neighbor: ' + str(neighbor))
                    # print('\t Others: ' + str([member for member in checkneighborlist if member is not neighbor]))
                    measure = self.__CheckNodes[checknodeid].getmessagetovar(varnode.getid())
                    weight = np.linalg.norm(measure, ord=1)
                    if weight != 0:
                        measure = measure / weight
                    else:
                        pass
                    varnode.setmessagefromcheck(checknodeid, measure)
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
        This method returns belief vector associated with variable node @var varnodeid.
        :param varnodeid: Identifier of variable node to be queried
        :return: Belief vector from variable node @var varnodeid
        """
        varnode = self.__VarNodes[varnodeid]
        return varnode.getestimate()

    def getestimates(self):
        """
        This method returns belief vectors for all variable nodes in bipartite graph.
        Belief vectors are sorted according to @var varnodeid.
        :return: Array of belief vectors from all variable nodes
        """
        estimates = np.empty((self.getvarcount(), self.getsparseseclength()), dtype=float)
        idx = 0
        for varnodeid in sorted(self.getvarlist()):
            estimates[idx] = self.__VarNodes[varnodeid].getestimate()
            idx = idx + 1
        return estimates

    def getextrinsicestimate(self, varnodeid):
        """
        This method returns belief vector associated with variable node @var varnodeid,
        based solely on extrinsic information.
        It does not incorporate information from local observation @var checknodeid = 0.
        :param varnodeid: Identifier of the variable node to be queried
        :return:
        """
        return self.__VarNodes[varnodeid].getmessagetocheck(0)


class SystematicEncoding(Graph):

    def __init__(self, check2varedges, infonodeindices, seclength):
        super().__init__(check2varedges, seclength)

        self.__InfoNodeIndices = sorted(infonodeindices)
        self.__InfoCount = len(self.__InfoNodeIndices)
        self.__ParityNodeIndices = [member for member in super().getvarlist() if member not in self.__InfoNodeIndices]
        self.__ParityCount = len(self.__ParityNodeIndices)
        print('Indices of information nodes: ' + str(self.__InfoNodeIndices))
        print('Indices of parity nodes: ' + str(self.__ParityNodeIndices))

    def getinfolist(self):
        return self.__InfoNodeIndices

    def getinfocount(self):
        return self.__InfoCount

    def getparitylist(self):
        return self.__ParityNodeIndices

    def getcodeword(self):
        """
        This method returns surviving codeword after systematic encoding and belief propagation.
        Codeword sections are sorted according to @var varnodeid.
        :return: Codeword in sections
        """
        codeword = np.empty((self.getvarcount(), self.getsparseseclength()), dtype=int)
        idx = 0
        for varnodeid in sorted(self.getvarlist()):
            block = np.zeros(self.getsparseseclength(), dtype=int)
            if np.max(self.getestimate(varnodeid)) > 0:
                block[np.argmax(self.getestimate(varnodeid))] = 1
            codeword[idx] = block
            idx = idx + 1
        return np.rint(codeword)

    def encodemessage(self, bits):
        """
        This method performs systematic encoding and belief propagation.
        Bipartite graph is initialized: local observations for information blocks are derived from message sequence,
        parity states are set to all ones.
        :param bits: Information bits comprising original message
        """
        if len(bits) == (self.getinfocount() * self.getseclength()):
            bits = np.array(bits).reshape((self.getinfocount(), self.getseclength()))
            # Container for fragmented message bits.
            bitsections = dict()
            idx = 0
            for varnodeid in self.getinfolist():
                # Message bits corresponding to fragment @var varnodeid.
                bitsections.update({varnodeid: bits[idx]})
                idx = idx + 1
            # Reinitialize factor graph to ensure there are no lingering states.
            # Node states are set to uninformative measures.
            self.reset()
            for varnodeid in self.getinfolist():
                # Compute index of fragment @var varnodeid
                fragment = np.inner(bitsections[varnodeid], 2 ** np.arange(self.getseclength()))
                # Set sparse representation to all zeros, except for proper location.
                sparsefragment = np.zeros(self.getsparseseclength(), dtype=int)
                sparsefragment[fragment] = 1
                # Set local observation for systematic variable nodes.
                self.setobservation(varnodeid, sparsefragment)
                # print('Variable node ' + str(varnodeid), end=' ')
                # print(' -- Observation changed to: ' + str(np.argmax(self.getobservation(varnodeid))))

            for idx in range(self.getvarcount()):  # Need to change this
                self.updatechecks()  # Update Check first
                self.updatevars()
                # Check if all variable nodes are set.
                if ((np.linalg.norm(np.rint(self.getestimates()).flatten(), ord=0)) == self.getvarcount()):
                    break
            codeword = np.rint(self.getestimates()).flatten()
            return codeword
        else:
            print('Length of input array is not ' + str(self.getinfocount() * self.getseclength()))

    def encodemessages(self, infoarray):
        codewords = []
        for messageindex in range(len(infoarray)):
            codewords.append(self.encodemessage(infoarray[messageindex]))
        return np.asarray(codewords)

    def encodesignal(self, infoarray):
        signal = [np.zeros(self.getsparseseclength(), dtype=float) for l in range(self.getvarcount())]
        for messageindex in range(len(infoarray)):
            signal = signal + self.encodemessage(infoarray[messageindex])
        return signal

    # Method testvalid needs attention; it may not be necessary
    def testvalid(self, codeword):  # ISSUE IN USING THIS FOR NOISY CODEWORDS, INPUT SHOULD BE MEASURE
        self.reset()
        if len(codeword) == (self.getvarcount() * self.getsparseseclength()):
            sparsesections = codeword.reshape((self.getvarcount(), self.getsparseseclength()))
            # Container for fragmented message bits.
            idx = 0
            for varnodeid in sorted(self.getvarlist()):
                # Sparse section corresponding to @var varnodeid.
                self.setobservation(varnodeid, sparsesections[idx])
                idx = idx + 1
                # print('Variable node ' + str(varnodeid), end=' ')
                # print(' -- Observation changed to: ' + str(np.argmax(self.getobservation(varnodeid))))

            for idx in range(self.getvarcount()): # NEEDS attention
                self.updatechecks()
                self.updatevars()
                # Check if all variable nodes remain sparse.
                if ((np.linalg.norm(np.rint(self.getestimates()).flatten(), ord=0)) == self.getvarcount()):
                    break
        else:
            print('Issue')
        return self.getcodeword()
