"""@package ccsfg

Package @package ccsfg contains the necessary building blocks to implement a bipartite factor graph tailored
to belief propagation.
The target appication is the coded compressed sensing, which necessitates the use of a large alphabet.
Thus, the structures of @class VariableNode and @class CheckNode assume that messages are passed using
 fast Fourier transform (FFT) techniques.
"""
import numpy as np
import scipy.linalg
import ffht


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


class VariableNode(GenericNode):
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
            # Normalize only if measure is not zero vector.
            # Numpy function np.isclose() breaks execution.
            weight = np.linalg.norm(measure, ord=1)
            if weight == 0:
                return measure
            else:
                # Under Numpy, division by zero seems to be a warning.
                return measure / np.linalg.norm(measure, ord=1)


class CheckNodeFFT(GenericNode):
    """
    Class @class CheckNode creates a single check node within bipartite factor graph.
    This class relies on fast Fourier transform.
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
        uninformative = np.fft.rfft(np.ones(self.__MessageLength, dtype=float))
        # The length of np.fft.rfft is NOT self.__MessageLength.
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


class CheckNodeFWHT(GenericNode):
    """
    Class @class CheckNode creates a single check node within bipartite factor graph.
    This class relies on fast Walsh-Hadamard transform.
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
        Reset every states check node to uninformative measures (FWHT of all ones)
        """
        uninformative = np.ones(self.__MessageLength, dtype=float)
        ffht.fht(uninformative) # @method fht acts in place on @type float
        for neighborid in self.getneighbors():
            self.setstate(neighborid, uninformative)

    def setmessagefromvar(self, varneighborid, message):
        """
        Incoming message from variable node neighbor @var vaneighborid to check node self.
        :param varneighborid: Variable node identifier of origin
        :param message: Incoming belief vector
        """
        message = message.astype(float)
        ffht.fht(message) # @method fht acts in place on @type float
        self.setstate(varneighborid, message)

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
                outgoing_fwht = states
            elif states.ndim == 2:
                try:
                    outgoing_fwht = np.prod(states, axis=0)
                except ValueError as e:
                    print(e)
                    return None
            else:
                raise RuntimeError('states.ndim = ' + str(np.array(states).ndim) + ' is not allowed.')
            outgoing = outgoing_fwht.astype(float)
            # Inversse of FWHT is, again, FWHT
            ffht.fht(outgoing) # @method fht acts in place on @type float
        # The outgoing message values should be indexed using the minus operation.
        # This is unnecessary over this particular field, since the minus is itself.
        return outgoing


class BipartiteGraph:
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
            self.__CheckNodes.update({checknodeid: CheckNodeFWHT(checknodeid, messagelength=self.__SparseSecLength)})
            # Add edges from check nodes to variable nodes.
            self.__CheckNodes[checknodeid].addneighbors(check2varedges[idx])
            # Create set of all variable node identifiers.
            self.__VarNodeIDs.update(check2varedges[idx])

        for varnodeid in self.__VarNodeIDs:
            # Create variable nodes and add them to dictionary @var self.__VariableNodes.
            self.__VarNodes[varnodeid] = VariableNode(varnodeid, messagelength=self.__SparseSecLength)

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
        return sorted(list(self.__VarNodes.keys()))

    def getvarcount(self):
        return len(self.__VarNodes)

    def getseclength(self):
        return self.__SecLength

    def getsparseseclength(self):
        return self.__SparseSecLength

    def getvarneighbors(self, varnodeid):
        return self.__VarNodes[varnodeid].getneighbors()

    def getcheckneighbors(self, checknodeid):
        return self.__CheckNodes[checknodeid].getneighbors()

    def getobservation(self, varnodeid):
        if varnodeid in self.getvarlist():
            return self.__VarNodes[varnodeid].getobservation()
        else:
            print('The retrival did not succeed.')
            print('Variable Node ID: ' + str(varnodeid))

    def getobservations(self):
        """
        This method returns local observations for all variable nodes in bipartite graph.
        Belief vectors are sorted according to @var varnodeid.
        :return: Array of local observations from all variable nodes
        """
        observations = np.empty((self.getvarcount(), self.getsparseseclength()), dtype=float)
        idx = 0
        for varnodeid in self.getvarlist():
            observations[idx] = self.__VarNodes[varnodeid].getobservation()
            idx = idx + 1
        return observations

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
            if np.isscalar(checknodelist):
                checknodelist = list([checknodelist])
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
            if np.isscalar(varnodelist):
                varnodelist = list([varnodelist])
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
                    measure = self.__CheckNodes[checknodeid].getmessagetovar(varnode.getid())
                    weight = np.linalg.norm(measure, ord=1)
                    if weight != 0:
                        measure = measure / weight
                    else:
                        pass
                    varnode.setmessagefromcheck(checknodeid, measure)
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
        for varnodeid in self.getvarlist():
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


class Encoding(BipartiteGraph):

    def __init__(self, check2varedges, infonodeindices, seclength):
        super().__init__(check2varedges, seclength)

        paritycheckmatrix = []
        for checknodeid in self.getchecklist():
            row = np.zeros(self.getvarcount(), dtype=int)
            for idx in self.getcheckneighbors(checknodeid):
                row[idx-1] = 1
            paritycheckmatrix.append(row)
        paritycheckmatrix = np.array(paritycheckmatrix)
        print('Size of parity check matrix: ' + str(paritycheckmatrix.shape))

        if infonodeindices is None:
            systematicmatrix = self.eliminationgf2(paritycheckmatrix)
            print(systematicmatrix)
            self.__paritycolindices = []
            paritynodeindices = []
            for idx in range(self.getcheckcount()):
                # Desirable indices are found in top rows of P_lu.transpose().
                # Below, columns of P_lu are employed instead of rows of P_lu.transpose().
                row = systematicmatrix[idx,:]
                jdx = np.argmax(row==1)
                self.__paritycolindices.append(jdx)
                paritynodeindices.append(jdx+1)
            self.__paritycolindices = sorted(self.__paritycolindices)
            self.__ParityNodeIndices = sorted(paritynodeindices)
            self.__ParityCount = len(self.__ParityNodeIndices)
            print('Number of parity column indices: ' + str(len(self.__paritycolindices)))

            self.__infocolindices = sorted([colidx for colidx in range(self.getvarcount()) if colidx not in self.__paritycolindices])
            infonodeindices = [varnodeid for varnodeid in self.getvarlist() if varnodeid not in paritynodeindices]
            self.__InfoNodeIndices = sorted(infonodeindices)
            self.__InfoCount = len(self.__InfoNodeIndices)
        else:
            self.__InfoNodeIndices = sorted(infonodeindices)
            self.__infocolindices = [idx - 1 for idx in self.__InfoNodeIndices]
            self.__InfoCount = len(self.__InfoNodeIndices)
            self.__ParityNodeIndices = [varnodeid for varnodeid in self.getvarlist() if varnodeid not in infonodeindices]
            self.__paritycolindices = [idx - 1 for idx in self.__ParityNodeIndices]
            self.__ParityCount = len(self.__ParityNodeIndices)

        print('Number of parity nodes: ' + str(len(set(self.__ParityNodeIndices))))
        self.__pc_parity = paritycheckmatrix[:, self.__paritycolindices]
        # print('Rank parity component: ' + str(np.linalg.matrix_rank(self.__pc_parity)))
        print(self.__pc_parity)

        print('Number of information nodes: ' + str(len(set(self.__InfoNodeIndices))))
        self.__pc_info = paritycheckmatrix[:, self.__infocolindices]
        # print('Rank info component: ' + str(np.linalg.matrix_rank(self.__pc_info)))
        print(self.__pc_info)

    def getinfolist(self):
        return self.__InfoNodeIndices

    def getinfocount(self):
        return self.__InfoCount

    def getparitylist(self):
        return self.__ParityNodeIndices

    def getparitycount(self):
        return self.__ParityCount

    def eliminationgf2(self, paritycheckmatrix):
        idx = 0
        jdx = 0
        while (idx < self.getcheckcount()) and (jdx < self.getvarcount()):
            # find value and index of largest element in remainder of column j
            while (np.amax(paritycheckmatrix[idx:, jdx]) == 0) and (jdx < self.getvarcount()):
                jdx += 1
            kdx = np.argmax(paritycheckmatrix[idx:, jdx]) + idx

            # Interchange rows @var kdx and @var idx
            row = np.copy(paritycheckmatrix[kdx])
            paritycheckmatrix[kdx] = paritycheckmatrix[idx]
            paritycheckmatrix[idx] = row

            rowidxtrailing = paritycheckmatrix[idx, jdx:]

            coljdx = np.copy(paritycheckmatrix[:, jdx]) #make a copy otherwise M will be directly affected
            coljdx[idx] = 0 #avoid xoring pivot @var rowidxtrailing with itself

            entries2flip = np.outer(coljdx, rowidxtrailing)
            # Python xor operator
            paritycheckmatrix[:, jdx:] = paritycheckmatrix[:, jdx:] ^ entries2flip

            idx += 1
            jdx +=1
        return paritycheckmatrix

    def getcodeword(self):
        """
        This method returns surviving codeword after systematic encoding and belief propagation.
        Codeword sections are sorted according to @var varnodeid.
        :return: Codeword in sections
        """
        codeword = np.empty((self.getvarcount(), self.getsparseseclength()), dtype=int)
        idx = 0
        for varnodeid in self.getvarlist():
            block = np.zeros(self.getsparseseclength(), dtype=int)
            if np.max(self.getestimate(varnodeid)) > 0:
                block[np.argmax(self.getestimate(varnodeid))] = 1
            codeword[idx] = block
            idx = idx + 1
        return np.rint(codeword)

    def encodemessage(self, bits):
        """
        This method performs encoding based on Gaussian elimination over GF2.
        :param bits: Information bits comprising original message
        """
        if len(bits) == (self.getinfocount() * self.getseclength()):
            bits = np.array(bits).reshape((self.getinfocount(), self.getseclength()))
            # Container for fragmented message bits.
            # Initialize variable nodes within information node indices
            codewordsparse = np.zeros((self.getvarcount(), self.getsparseseclength()))
            for idx in range(self.getinfocount()):
                # Compute index of fragment @var varnodeid
                fragment = np.inner(bits[idx], 2 ** np.arange(self.getseclength())[::-1])
                # Set sparse representation to all zeros, except for proper location.
                sparsefragment = np.zeros(self.getsparseseclength(), dtype=int)
                sparsefragment[fragment] = 1
                # Add sparse section to codeword.
                codewordsparse[self.__infocolindices[idx]] = sparsefragment
            for idx in range(self.getparitycount()):
                parity = np.remainder(self.__pc_info[idx,:] @ bits, 2)
                fragment = np.inner(parity, 2 ** np.arange(self.getseclength())[::-1])
                # Set sparse representation to all zeros, except for proper location.
                sparsefragment = np.zeros(self.getsparseseclength(), dtype=int)
                sparsefragment[fragment] = 1
                # Add sparse section to codeword.
                codewordsparse[self.__paritycolindices[idx]] = sparsefragment
            codeword = np.array(codewordsparse).flatten()
            return codeword
        else:
            print('Length of input array is not ' + str(self.getinfocount() * self.getseclength()))
            print('Number of information sections is ' + str(self.getinfocount()))

    def encodemessageBP(self, bits):
        """
        This method performs systematic encoding through belief propagation.
        Bipartite graph is initialized: local observations for information blocks are derived from message sequence,
        parity states are set to all ones.
        :param bits: Information bits comprising original message
        """
        if len(bits) == (self.getinfocount() * self.getseclength()):
            bits = np.array(bits).reshape((self.getinfocount(), self.getseclength()))
            # Container for fragmented message bits.
            bitsections = dict()
            # Reinitialize factor graph to ensure there are no lingering states.
            # Node states are set to uninformative measures.
            self.reset()
            idx = 0
            # Initialize variable nodes within information node indices
            for varnodeid in self.getinfolist():
                # Message bits corresponding to fragment @var varnodeid.
                bitsections.update({varnodeid: bits[idx]})
                idx = idx + 1
                # Compute index of fragment @var varnodeid
                fragment = np.inner(bitsections[varnodeid], 2 ** np.arange(self.getseclength())[::-1])
                # Set sparse representation to all zeros, except for proper location.
                sparsefragment = np.zeros(self.getsparseseclength(), dtype=int)
                sparsefragment[fragment] = 1
                # Set local observation for systematic variable nodes.
                self.setobservation(varnodeid, sparsefragment)
                # print('Variable node ' + str(varnodeid), end=' ')
                # print(' -- Observation changed to: ' + str(np.argmax(self.getobservation(varnodeid))))

            # Start with full list of check nodes to update.
            checknodes2update = set(self.getchecklist())
            self.updatechecks(checknodes2update)
            # Start with list of parity variable nodes to update.
            varnodes2update = set(self.getvarlist())
            self.updatevars(varnodes2update)
            # The number of parity variable nodes acts as upper bound.
            for iteration in range(self.getparitycount()):
                checkneighbors = set()
                varneighbors = set()

                # Update check nodes and check for convergence
                self.updatechecks(checknodes2update)
                for checknodeid in checknodes2update:
                    if set(self.getcheckneighbors(checknodeid)).isdisjoint(varnodes2update):
                        checknodes2update = checknodes2update - {checknodeid}
                        varneighbors.update(self.getcheckneighbors(checknodeid))
                if varneighbors != set():
                    self.updatevars(varneighbors)

                # Update variable nodes and check for convergence
                self.updatevars(varnodes2update)
                for varnodeid in varnodes2update:
                    currentmeasure = self.getestimate(varnodeid)
                    currentweight1 = np.linalg.norm(currentmeasure, ord=1)
                    if currentweight1 == 1:
                        varnodes2update = varnodes2update - {varnodeid}
                        checkneighbors.update(self.getvarneighbors(varnodeid))
                if checkneighbors != set():
                    self.updatechecks(checkneighbors)

                if np.array_equal(np.linalg.norm(np.rint(self.getestimates()), ord=0, axis=1), [1] * self.getvarcount()):
                    break

            self.updatechecks()
            # print(np.linalg.norm(np.rint(self.getestimates()), ord=0, axis=1))
            codeword = np.rint(self.getestimates()).flatten()
            return codeword
        else:
            print('Length of input array is not ' + str(self.getinfocount() * self.getseclength()))

    def encodemessages(self, infoarray):
        """
        This method encodes multiple messages into codewords by performing systematic encoding
        and belief propagation on each individual message.
        :param infoarray: array of binary messages to be encoded
        """
        codewords = []
        for messageindex in range(len(infoarray)):
            codewords.append(self.encodemessage(infoarray[messageindex]))
        return np.asarray(codewords)

    def encodesignal(self, infoarray):
        """
        This method encodes multiple messages into a signal
        :param infoarray: array of binary messages to be encoded
        """
        signal = np.array([np.zeros(self.getsparseseclength(), dtype=float) for l in range(self.getvarcount())]).flatten()
        for messageindex in range(len(infoarray)):
            signal = signal + self.encodemessage(infoarray[messageindex])
        return signal

    # The @method testvalid needs attention; it may not be necessary
    def testvalid(self, codeword):  # ISSUE IN USING THIS FOR NOISY CODEWORDS, INPUT SHOULD BE MEASURE
        # Reinitialize factor graph to ensure there are no lingering states.
        # Node states are set to uninformative measures.
        self.reset()
        if (len(codeword) == (self.getvarcount() * self.getsparseseclength())) and (
                np.linalg.norm(codeword, ord=0) == self.getvarcount()):
            sparsesections = codeword.reshape((self.getvarcount(), self.getsparseseclength()))
            # Container for fragmented message bits.
            idx = 0
            for varnodeid in self.getvarlist():
                # Sparse section corresponding to @var varnodeid.
                self.setobservation(varnodeid, sparsesections[idx])
                idx = idx + 1
                # print('Variable node ' + str(varnodeid), end=' ')
                # print(' -- Observation changed to: ' + str(np.argmax(self.getobservation(varnodeid))))

            # Check if all variable nodes remain sparse.
            self.updatechecks()
            self.updatevars()
            if np.array_equal(np.rint(self.getestimates()).flatten(), self.getobservations().flatten()):
                # print('Codeword is consistent.')
                return self.getcodeword()
            else:
                print('Codeword has issues.')
                print(np.sum(self.getobservations() - self.getestimates(), axis=1))
        else:
            print(np.linalg.norm(np.rint(self.getestimates()).flatten(), ord=0))
            print(np.linalg.norm(self.getobservations().flatten(), ord=0))
            print('Codeword has issues.')
