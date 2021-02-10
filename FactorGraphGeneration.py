__author__ = 'JF Chamberland'

import ccsfg as ccsfg
import numpy as np


class GraphTest(ccsfg.SystematicEncoding):
    def __init__(self, seclength=2):
        self.__Check2VarEdges = []
        self.__Check2VarEdges.append([1, 2, 3])
        self.__Check2VarEdges.append([2, 4])
        super().__init__(self.__Check2VarEdges, [1, 2], seclength)
        self.__DepthFromRoot = 8  # CHECK

    def getmaxdepth(self):
        return self.__DepthFromRoot


class Graph8(ccsfg.SystematicEncoding):

    def __init__(self, seclength=16):
        self.__Check2VarEdges = []
        self.__Check2VarEdges.append([1, 2, 3])
        self.__Check2VarEdges.append([4, 5, 6])
        self.__Check2VarEdges.append([7, 8, 9])
        self.__Check2VarEdges.append([10, 11, 12])
        self.__Check2VarEdges.append([1, 7, 13])
        self.__Check2VarEdges.append([2, 10, 14])
        self.__Check2VarEdges.append([4, 8, 15])
        self.__Check2VarEdges.append([5, 11, 16])
        super().__init__(self.__Check2VarEdges, [1, 2, 4, 5, 7, 8, 10, 11], seclength)
        self.__DepthFromRoot = 32  # CHECK

    def getmaxdepth(self):
        return self.__DepthFromRoot


class Graph6(ccsfg.SystematicEncoding):

    def __init__(self, seclength=16):
        self.__Check2VarEdges = []
        self.__Check2VarEdges.append([1, 2, 7])
        self.__Check2VarEdges.append([1, 3, 8])
        self.__Check2VarEdges.append([7, 4, 9])
        self.__Check2VarEdges.append([1, 12])
        self.__Check2VarEdges.append([2, 5, 11])
        self.__Check2VarEdges.append([2, 6, 10])
        self.__Check2VarEdges.append([3, 13])
        self.__Check2VarEdges.append([4, 14])
        self.__Check2VarEdges.append([5, 15])
        self.__Check2VarEdges.append([6, 16])
        super().__init__(self.__Check2VarEdges, [1, 2, 3, 4, 5, 6], seclength)
        self.__DepthFromRoot = 32  # CHECK

    def getmaxdepth(self):
        return self.__DepthFromRoot


class Graph62(ccsfg.SystematicEncoding):

    def __init__(self, seclength=16):
        self.__Check2VarEdges = []
        self.__Check2VarEdges.append([1, 2, 3])
        self.__Check2VarEdges.append([4, 5, 6])
        self.__Check2VarEdges.append([7, 8, 9])
        self.__Check2VarEdges.append([1, 7, 10])
        self.__Check2VarEdges.append([2, 5, 11])
        self.__Check2VarEdges.append([4, 8, 12])
        self.__Check2VarEdges.append([3, 12, 13])
        self.__Check2VarEdges.append([6, 10, 14])
        self.__Check2VarEdges.append([9, 11, 15])
        self.__Check2VarEdges.append([13, 14, 15, 16])
        super().__init__(self.__Check2VarEdges, [1, 2, 4, 5, 7, 8], seclength)
        self.__DepthFromRoot = 16  # CHECK

    def getmaxdepth(self):
        return self.__DepthFromRoot


def decoder(graph, stateestimates, count):  # NEED ORDER OUTPUT IN LIKELIHOOD MAYBE
    """
    This method seeks to disambiguate codewords from node states.
    Gather local state estimates from variables nodes and retain top values in place.
    Set values of other indices within every section to zero.
    Perform belief propagation and return `count` likely codewords.
    :param graph: Bipartite graph for error correcting code.
    :param stateestimates: Local estimates from variable nodes.
    :param count: Maximum number of codewords returned.
    :return: List of likely codewords.
    """

    # Resize @var stateestimates to match local measures from variable nodes.
    stateestimates = np.resize(stateestimates, 
                        (graph.getvarcount(), graph.getsparseseclength())).astype(float)
    stateestimates = stateestimates / np.sum(stateestimates, axis=1).reshape((-1, 1))   # normalize so that each measure is a valid PMF
    recoveredcodewords = []                                                             # list of candidate codewords
    trailingtopindices = list(np.argpartition(stateestimates[0, :], -count)[-count:])   # indices of `count` most likely roots
    print('\nState Estimates: \n' + str(stateestimates))
    print('Trailing Top Indices: \n' + str(trailingtopindices))
    
    # Iterate through evey retained location in root section
    for topidx in trailingtopindices:

        # Reset graph, including check nodes. This is critical for every root location.
        graph.reset()
        print('Root section ID: ' + str(topidx))

        # If a topidx corresponds to a 0 in stateestimates, it has 0 probability...
        if stateestimates[0, topidx] == 0:
            print('Skipping root index %d because it has zero probability.' % topidx)
            continue

        # Initialize graph.  Set belief measure for 1st node to be an standard basis vector of message root index
        rootsingleton = np.zeros(graph.getsparseseclength())
        rootsingleton[topidx] = 1
        graph.setobservation(1, rootsingleton)
        for idx in range(1, graph.getvarcount()):
            graph.setobservation(idx + 1, stateestimates[idx, :])

        # Start with full list of nodes to update.
        graph.updatechecks()
        graph.updatevars()
        varnodes2update = set(graph.getvarlist())
        checknodes2update = set(graph.getchecklist())

        # run maxdepth iterations of BP
        for iteration in range(graph.getmaxdepth()):  # Max depth
            
            # Update check and variable node messages; Update Checks first
            graph.updatechecks(checknodes2update)  
            graph.updatevars(varnodes2update)

            # prune sets of nodes to update in each iteration
            for varnodeid in varnodes2update:
                weight = graph.getestimate(varnodeid)
                if np.sum(weight) == np.amax(weight):
                    varnodes2update = varnodes2update - {varnodeid}
                    # print('Variable nodes to update: ' + str(varnodes2update))

            for checknodeid in checknodes2update:
                if set(graph.getcheckneighbors(checknodeid)).isdisjoint(varnodes2update):
                    checknodes2update = checknodes2update - {checknodeid}
                    # print('Check nodes to update: ' + str(checknodes2update))

        # recover estimated codeword
        decoded = graph.getcodeword().flatten()
        decodedsum = np.sum(decoded.flatten())

        # Case 1: decoder successfully disambiguates a codeword starting from given root node
        if (decodedsum == graph.getvarcount()) and np.array_equal(decoded, graph.testvalid(decoded).flatten()): 
            
            # add estimated codeword to list of recovered codewords
            recoveredcodewords.append(decoded)
            print('Valid codeword added to list of recovered codewords.')

            # remove belief associated with disambiguated codeword from stateestimate
            idxdecoded = np.mod(np.where(decoded.flatten())[0], graph.getsparseseclength())
            for i in range(graph.getvarcount()):
                if (stateestimates[i, idxdecoded[i]] >= 1/count):
                    stateestimates[i, idxdecoded[i]] -= 1/count
                else:
                    stateestimates[i, idxdecoded[i]] /= count

            # If there is still significant belief on this root, try searching for another codeword that begins with this root
            if (stateestimates[0, idxdecoded[0]] > 1/(2*count)) and (trailingtopindices.count(topidx) < count):
                print('Root %d has potential for another valid codeword.  Adding to the queue for retrial. ' % topidx)
                trailingtopindices.append(topidx)

        # Case 2: JF gives the decoder a weird input.   Not sure how this would happen...
        elif decodedsum > graph.getvarcount(): 
            print('Disambiguation failed.')

        # Case 3: decoder fails to disambiguate a valid codeword starting from given root node
        else:
            # NOTE: The occurence of this event offers us some important information.  One of the following errors must have happened:  
            # 1. One of the most likely root nodes (stored in trailingtopindices) is the root of more than one valid codeword, 
            #    each of which is equally likely.  This forces BP to converge to something that is not a valid codeword.
            # 2. Our estimate of the most likely root nodes (trailingtopindices) excluded a valid root node in favor of an invalid root node.
            # 3. We zeroed out a necessary component of one of the codewords, thus making it impossible for BP to find the codeword

            print('No consistent codeword found.')

            # Solution to problem # 1:
            # Retry disambiguating a valid codeword starting from the root node after decreasing the belief on the false codeword indices.
            # This will, with some probability, force the two valid codewords to have unequal probabilities, enabling BP to converge to the
            # more likely codeword correctly. 
            #
            # Try again starting from this root node 
            if trailingtopindices.count(topidx) < count:
                trailingtopindices.append(topidx)
                print('Adding root node %d to the queue for retrial.' % topidx)

            # Decrease belief on invalid codeword indices
            idxdecoded = np.mod(np.where(decoded.flatten())[0], graph.getsparseseclength())
            if idxdecoded.size > 0:
                for i in range(graph.getvarcount()):            # FIXME: potential problem if one of the sections is all zeros and has no 1s...
                    stateestimates[i, idxdecoded[i]] *= 0.9

            # Solution to problem #2: 
            # (Not yet implemented).  We would need to keep the delta*count, delta > 1, top root indices and try to 
            # disambiguate a message starting from each of these root indices.  This should be a simple modification to this code,
            # should we deem this modification necessary. 

            # Solution to problem #3: 
            # In the current implementation of the decoder, this should not happen. 

    # Order candidates
    likelihoods = []
    for candidate in recoveredcodewords:
        isolatedvalues = np.prod((candidate, stateestimates.flatten()), axis=0)
        isolatedvalues.resize(graph.getvarcount(), graph.getsparseseclength())
        likelihoods.append(np.prod(np.amax(isolatedvalues, axis=1)))
    idxsorted = np.argsort(likelihoods)
    recoveredcodewords = [recoveredcodewords[idx] for idx in idxsorted[::-1]]
    return np.array(recoveredcodewords)


def numbermatches(codewords, recoveredcodewords, maxcount=None):
    """
    Counts number of matches between `codewords` and `recoveredcodewords`.
    CHECK: Does not ensure uniqueness.
    :param codewords: List of true codewords.
    :param recoveredcodewords: List of candidate codewords from most to least likely.
    :return: Number of true codewords recovered.
    """
    # Provision for scenario where candidate count is smaller than codeword count.
    if maxcount is None:
        maxcount = min(len(codewords), len(recoveredcodewords))
    else:
        maxcount = min(len(codewords), len(recoveredcodewords), maxcount)
    matchcount = 0
    for candidateindex in range(maxcount):
        candidate = recoveredcodewords[candidateindex]
        # print('Candidate codeword: ' + str(candidate))
        # print(np.equal(codewords,candidate).all(axis=1)) # Check if candidate individual codewords
        matchcount = matchcount + (np.equal(codewords, candidate).all(axis=1).any()).astype(int)  # Check if matches any
    return matchcount


def displayinfo(graph, binarysequence):
    alphabet = graph.getseclength()
    binsections = binarysequence.reshape(alphabet, -1)
    sections = []
    # for sectionindex in range(len(binarysequence)):
    # sections.append(np.sum([2**i - 1 for i in np.argmax(binsections[sectionindex])-1)
    print(sections)


# Simulation Parameters
# infoarray = [[1, 0, 0, 0, 1, 0]]
# NumberDevices = 1

# # Instantiate outer code
# TestCode = GraphTest(3)
# TestCode.reset()
# # TestCode.printgraph()


# codeword = TestCode.encodemessage(infoarray[0])  # message is now outer-encoded
# print('Information bits:\n' + str(infoarray))
# print('Signal sections:\n' + str(codeword))
# TestCode.printgraphcontent()

# signal = TestCode.encodesignal(infoarray)
# print('Signal reshaped:\n' + str(signal))


# testvector = np.ones((TestCode.getvarcount(),TestCode.getsparseseclength()))
# testvector[0] = np.zeros(TestCode.getsparseseclength())
# for sectionid in TestCode.getvarlist():
#     TestCode.setobservation(sectionid, testvector[sectionid-1,:])
#     print(np.sum([TestCode.getextrinsicestimate(varnodeid).flatten() for varnodeid in TestCode.getvarlist()],axis=1))
# for iteration in range(TestCode.getmaxdepth()):    # Max depth
#     TestCode.updatechecks()
#     TestCode.updatevars()
#     print(np.sum([TestCode.getextrinsicestimate(varnodeid) for varnodeid in TestCode.getvarlist()],axis=1))

# raise Exception('Stopping')


OuterCode = GraphTest(3)
OuterCode.printgraph()

NumberDevices = 3
# infoarray = [[1, 0, 0, 0, 1, 0], [0, 1, 0, 1, 1, 1], [0, 0, 1, 1, 0, 0]]          # Easy test case
# infoarray = [[1, 0, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0], [0, 1, 1, 0, 1, 0]]          # Hard test case: lots of collisions
infoarray = [list(np.random.randint(2, size=(2*3))) for i in range(NumberDevices)]  # Random test case
print(infoarray)

NumberDevices = len(infoarray)

codewords = OuterCode.encodemessages(infoarray)
# print('Codewords shape: ' + str(codewords.shape))
codeword = codewords[0]
# print('Codeword shape' + str(codeword.shape))
notcodeword = codeword[::-1]
signal = OuterCode.encodesignal(infoarray)
print('Signal: ' + str(signal))
print('\n')

# output = OuterCode.testvalid(notcodeword.flatten())
# print('Test codeword: ' + str(np.linalg.norm(output.flatten(),ord=1)))
# print(output)

# print('Estimates shape' + str(OuterCode.getestimates().shape))
# print(np.linalg.norm(OuterCode.getestimates().flatten(),ord=1))
# print(np.linalg.norm(output.flatten() - codeword,ord=1))

originallist = codewords.copy()
recoveredcodewords = decoder(OuterCode,signal,len(infoarray))
print('\nOriginal List: \n' + str(originallist))
print('Recovered codewords: \n' + str(recoveredcodewords))

matches = numbermatches(originallist,recoveredcodewords)
print('Matches: %d. Percent Recovered: %3.2f%%' % (matches, 100*matches/NumberDevices))