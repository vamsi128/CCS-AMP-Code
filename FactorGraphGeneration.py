__author__ = 'JF Chamberland'

import FactorGraph as FG
import numpy as np

class Graph8(FG.Graph):

    def __init__(self,seclength=16):
        self.__Check2VarEdges = [[0]]   # Check node 0 is a dummy node used for indexing.
        self.__Check2VarEdges.append([1, 2, 3])
        self.__Check2VarEdges.append([4, 5, 6])
        self.__Check2VarEdges.append([7, 8, 9])
        self.__Check2VarEdges.append([10, 11, 12])
        self.__Check2VarEdges.append([1, 7, 13])
        self.__Check2VarEdges.append([2, 10, 14])
        self.__Check2VarEdges.append([4, 8, 15])
        self.__Check2VarEdges.append([5, 11, 16])
        # self.__UniqueVarCount = len(set([varnodeid for edges in self.__Check2VarEdges for varnodeid in edges]))-1
        self.__UniqueVarCount = 16
        super().__init__(self.__Check2VarEdges, self.__UniqueVarCount, [1, 2, 4, 5, 7, 8, 10, 11], seclength)
        self.__DepthFromRoot = self.__UniqueVarCount # CHECK

    def getmaxdepth(self):
        return self.__DepthFromRoot


class Graph6(FG.Graph):

    def __init__(self,seclength=16):
        self.__Check2VarEdges = [[0]]   # Check node 0 is a dummy node used for indexing.
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
        # self.__UniqueVarCount = len(set([varnodeid for edges in self.__Check2VarEdges for varnodeid in edges]))-1
        self.__UniqueVarCount = 16
        super().__init__(self.__Check2VarEdges, self.__UniqueVarCount, [1, 2, 3, 4, 5, 6], seclength)
        self.__DepthFromRoot = 2*self.__UniqueVarCount # CHECK

    def getmaxdepth(self):
        return self.__DepthFromRoot


class Graph62(FG.Graph):

    def __init__(self,seclength=16):
        self.__Check2VarEdges = [[0]]   # Check node 0 is a dummy node used for indexing.
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
        # self.__UniqueVarCount = len(set([varnodeid for edges in self.__Check2VarEdges for varnodeid in edges]))-1
        self.__UniqueVarCount = 16
        super().__init__(self.__Check2VarEdges, self.__UniqueVarCount, [1, 2, 4, 5, 7, 8], seclength)
        self.__DepthFromRoot = 2*self.__UniqueVarCount # CHECK

    def getmaxdepth(self):
        return self.__DepthFromRoot


def decoder(graph, stateestimates, count): # NEED ORDER OUTPUT IN LIKELIHOOD MAYBE
    """
    Takes state estimates and return `count` likely codewords.
    :param graph: Bipartite graph for error correcting code.
    :param stateestimates: Local estimates for variable nodes.
    :param count: Maximum number of codewords returned.
    :return: List of likely codewords.
    """

    stateestimates.resize(graph.getvarcount(), graph.getsparseseclength())
    thresholdedestimates = np.zeros(stateestimates.shape)
    # hardestimates = np.zeros(stateestimates.shape)


    # NOTE: the pruning of impossible paths prior to root decoding doesn't seem to help.
    # topindices = []
    # graph.reset()
    # idx: int
    # for idx in range(graph.getvarcount()):
    #     vector = stateestimates[idx,:].copy()
    #     trailingtopindices = np.argpartition(vector, -128)[-128:]
    #     topindices.append(trailingtopindices)    # Indices of `count` most likely locations
    #     for topidx in topindices[idx]:    # Set most likely locations to one
    #         hardestimates[idx,topidx] = 1 if (vector[topidx] != 0) else 0
    #     print(np.linalg.norm(hardestimates[idx,:], ord=0), end=' ')
    #     graph.setobservation(idx+1,hardestimates[idx,:])
    # print('\n')
    #
    # for iter in range(16):    # For Graph6
    #     graph.updatechecks()  # Update Check first
    #     graph.updatevars()
    #
    # for idx in range(graph.getvarcount()):
    #     hardestimates[idx] = graph.getestimate(idx+1)
    #     print(np.linalg.norm(hardestimates[idx,:], ord=0), end=' ')
    #     vector = hardestimates[idx,:].copy()
    #     for topidx in topindices[idx]:    # Set most likely locations to one
    #         thresholdedestimates[idx,topidx] = stateestimates[idx,topidx] if (vector[topidx] != 0) else 0
    # print('\n')



    # Find `count` most likely locations in every section and zero out the rest.
    topindices = []
    idx: int
    for idx in range(graph.getvarcount()):
        vector = stateestimates[idx,:].copy()
        trailingtopindices = np.argpartition(vector, -1024)[-1024:]
        topindices.append(trailingtopindices)    # Indices of `count` most likely locations
        for topidx in topindices[idx]:    # Set most likely locations to one
            thresholdedestimates[idx,topidx] = vector[topidx] if (vector[topidx] != 0) else 0
        # print(np.linalg.norm(thresholdedestimates[idx,:], ord=0), end=' ')


    recoveredcodewords = []
    vector = thresholdedestimates[0,:].copy()    # Section 0 acts as root
    topindices = np.argpartition(vector, -count)    # Indices of `count` most likely locations in root
    for topidx in topindices[-count:]:    # Iterating through evey location in root node
        print('Root section ID: ' + str(topidx))
        graph.reset()   # Need reset for decoding every root location
        newvector = np.zeros(graph.getsparseseclength())
        newvector[topidx] = 1 if (vector[topidx] != 0) else 0
        graph.setobservation(1, newvector)
        for idx in range(1,graph.getvarcount()):
            graph.setobservation(idx+1,stateestimates[idx,:])
            # graph.setobservation(idx+1,thresholdedestimates[idx,:])



        ## This may only work for hierchical settings.

        varnodes2update = set(graph.getvarlist())
        checknodes2update = set(graph.getchecklist())
        for iter in range(graph.getmaxdepth()):    # Max depth
            graph.updatechecks(checknodes2update)  # Update Check first
            graph.updatevars(varnodes2update)
            for varnodeid in varnodes2update:
                weight = graph.getestimate(varnodeid)
                if np.sum(weight) == np.amax(weight):
                    varnodes2update = varnodes2update - set([varnodeid])
                    # print('Variable nodes to update: ' + str(varnodes2update))
            for checknodeid in checknodes2update:
                if set(graph.getcheckneighbors(checknodeid)).isdisjoint(varnodes2update):
                    checknodes2update = checknodes2update - set([checknodeid])
                    # print('Check nodes to update: ' + str(checknodes2update))




        #
        # for iter in range(graph.getmaxdepth()):    # Max depth
        #     graph.updatechecks()
        #     graph.updatevars()

        decoded = graph.getcodeword().flatten()
        decodedsum = np.sum(decoded.flatten())
        if decodedsum == graph.getvarcount():
            recoveredcodewords.append(decoded)
        elif decodedsum > graph.getvarcount(): # CHECK: Can be improved later
            print('Disambiguation failed.')
            recoveredcodewords.append(decoded)
        else:
            pass

    # Order candidates
    likelihoods = []
    for candidate in recoveredcodewords:
        isolatedvalues = np.prod((candidate,stateestimates.flatten()),axis=0)
        isolatedvalues.resize(graph.getvarcount(), graph.getsparseseclength())
        likelihoods.append(np.prod(np.amax(isolatedvalues,axis=1)))
    idxsorted = np.argsort(likelihoods)
    recoveredcodewords = [recoveredcodewords[idx] for idx in idxsorted[::-1]]
    return recoveredcodewords

def numbermatches(codewords, recoveredcodewords, maxcount=None):
    """
    Counts number of matches between `codewords` and `recoveredcodewords`.
    CHECK: Does not insure uniqueness.
    :param codewords: List of true codewords.
    :param recoveredcodewords: List of candidate codewords from most to least likely.
    :return: Number of true codewords recovered.
    """
    # Provision for scenario where candidate count is smaller than codeword count.
    if maxcount is None:
        maxcount = min(len(codewords),len(recoveredcodewords))
    else:
        maxcount = min(len(codewords),len(recoveredcodewords),maxcount)
    matchcount = 0
    for candidateindex in range(maxcount):
        candidate = recoveredcodewords[candidateindex]
        # print('Candidate codeword: ' + str(candidate))
        # print(np.equal(codewords,candidate).all(axis=1)) # Check if candidate individual codewords
        matchcount = matchcount + (np.equal(codewords,candidate).all(axis=1).any()).astype(int)   # Check if matches any
    return matchcount


def displayinfo(graph, binarysequence):
    alphabet = graph.getseclength()
    binsections = binarysequence.reshape(alphabet,-1)
    sections = []
    # for sectionindex in range(len(binarysequence)):
        # sections.append(np.sum([2**i - 1 for i in np.argmax(binsections[sectionindex])-1)
    print(sections)



#
#
#
# TestCode = Graph8(8)
# TestCode.reset()
# testvector = np.ones((TestCode.getvarcount(),TestCode.getsparseseclength()))
# testvector[0] = np.zeros(TestCode.getsparseseclength())
# for sectionid in TestCode.getvarlist():
#     TestCode.setobservation(sectionid, testvector[sectionid-1,:])
#     print(np.sum([TestCode.getextrinsicestimate(varnodeid) for varnodeid in TestCode.getvarlist()],axis=1))
# for iter in range(TestCode.getmaxdepth()):    # Max depth
#     TestCode.updatechecks()
#     TestCode.updatevars()
#     print(np.sum([TestCode.getextrinsicestimate(varnodeid) for varnodeid in TestCode.getvarlist()],axis=1))


#
# OuterCode = Graph6(4)
# OuterCode.printgraph()
# NumberDevices = 1
#
# infoarray = np.random.randint(2, size=(NumberDevices,OuterCode.getinfocount()*OuterCode.getseclength()))
# print('Information bits: ' + str(displaycodeword(OuterCode,infoarray)))
# # print('Signal shape: ' + str(OuterCode.encodesignal(infoarray)))
# signal = OuterCode.encodesignal(infoarray).reshape(1, -1)
# print('Signal reshaped: ' + str(signal.shape))
#
#
# codewords = OuterCode.encodemessages(infoarray)
# print('Codewords shape: ' + str(codewords.shape))
# codeword = codewords[0]
# print('Codeword shape' + str(codeword.shape))
# notcodeword = codeword[::-1]
#
# output = OuterCode.testvalid(notcodeword.flatten())
# print('Test codeword: ' + str(np.linalg.norm(output.flatten(),ord=1)))
# print(output)
#
# # print('Estimates shape' + str(OuterCode.getestimates().shape))
# # print(np.linalg.norm(OuterCode.getestimates().flatten(),ord=1))
# # print(np.linalg.norm(output.flatten() - codeword,ord=1))
#
# originallist = codewords.copy()
# recoveredcodewords = decoder(OuterCode,signal[::-1],NumberDevices)
# print(recoveredcodewords)
#
# matches = numbermatches(originallist,recoveredcodewords)
# print(matches)

