import numpy as np
import math
import time
import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import TwoSlopeNorm
from scipy import sparse
import seaborn as sns
from sympy.combinatorics import Permutation, PermutationGroup

import netket as nk
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import json

PATH = os.getcwd()
start = time.time()

def generateStateTable(n,n0,N): # assumes ab > n0 > 0 and N = n choose n0
    states = []
    state = np.concatenate((np.ones(n-n0),-1*np.ones(n0))).astype(int)
    for i in range(0,N):
        states.append(np.copy(state))
        j = 0
        flag = True
        flip_count = 0
        up_count = 0
        while (flag): 
            if (j == n-1):
                for m in range(0,n):
                    if (state[m] != 1 and state[m+1] == 1):
                        flip_count += 1
                if (flip_count == 1):
                    break
            if (state[j] != 1):
                j += 1  
            elif (state[j+1] != 1):
                for m in range(0,j):
                    if (state[m] != 1 and state[m+1] == 1):
                        flip_count += 1
                    if (state[m] == 1):
                        up_count += 1
                if (flip_count == 1):
                    state[j],state[j+1] = state[j+1],state[j]
                    for k in range(1,up_count+1):
                        state[j-k],state[k-1] = state[k-1],state[j-k]
                else:
                    state[j],state[j+1] = state[j+1],state[j]
                flag = False
            else:
                j += 1
    return states

class Node:
    def __init__(self, position, xadj, yadj):
        self.position = position
        self.xadj = xadj
        self.yadj = yadj
    
    def __str__(self):
        return f"{self.position} {self.xadj} {self.yadj}"
    
def squareAdjacencyList(a,b): # constructs a periodic adjacency graph with width a and height b
    nodes = []
    
    for i in range(0,a):
        for j in range(0,b):
            xadj = [[(i-1) % a,j],[(i+1) % a,j]]
            yadj = [[i,(j-1) % b],[i,(j+1) % b]]
            nodes.append(Node([[i,j]],xadj,yadj))

    return nodes

def firstneighbors(a,b): 
    nodes = squareAdjacencyList(a,b)
    N = a*b
    J = [[0 for col in range(N)] for row in range(N)]

    for i in range(0,N-1):
        for j in range(i+1,N):
            flag = False
            for xptr in nodes[i].xadj:
                if xptr in nodes[j].position:
                    flag = True
            for yptr in nodes[i].yadj:
                if yptr in nodes[j].position:
                    flag = True
            if flag:
                J[i][j] = 1
                J[j][i] = 1
        
    return np.array(J)

def secondneighbors(a,b):
    nodes = squareAdjacencyList(a,b)
    N = a*b
    J = [[0 for col in range(N)] for row in range(N)]

    for i in range(0,N-1):
        for j in range(i+1,N):
            flag = False

            for xptr in nodes[i].xadj:
                try:
                    for k in range(0,N):
                        if xptr in nodes[k].position:
                            intermediate = k
                    for yptr in nodes[intermediate].yadj:
                        if yptr in nodes[j].position:
                            flag = True
                except UnboundLocalError:
                    pass

            if flag:
                J[i][j] = 1
                J[j][i] = 1

    return np.array(J)

a = 6 # x-range of supercell
b = 6 # y-range of supercell
N1 = firstneighbors(a,b)
N2 = secondneighbors(a,b)
n = a*b # number of sites in lattice

n0 = n // 2 # number of down spins in the string (taken as floor(n/2))
N = int(math.factorial(n)/(math.factorial(n0)*math.factorial(n-n0))) # number of states

stateTable = generateStateTable(n,n0,N)

def stateValBin(state):
    sum = 0
    for i in range(len(state)):
        if state[i] == 1:
            sum += 2**i
    return sum

def searchState(state):
    high = N-1
    low = 0
    while True:
        mid = math.floor((high+low)/2)
        if np.array_equal(state,stateTable[mid]):
            return mid
        elif stateValBin(state) > stateValBin(stateTable[mid]):
            low = mid+1
        else:
            high = mid-1

def transSym(state,x,y): # translation vector (x,y)
    newState = []
    for j in range(b):
        for i in range(a):
            ii = (i-x) % a
            jj = (j-y) % b 
            newState.append(state[a*jj+ii])
    return np.array(newState)

def spinFlipSym(state):
    return -1*state

def rotSiteSym(state,n): # rotate about the origin by n\pi/2 radians CCW
    newState = []
    if n % 4 == 0:
        return state
    elif n % 4 == 1:
        for j in range(b):
            for i in range(a):
                ii = j % a
                jj = (-i) % b
                newState.append(state[a*jj+ii])
    elif n % 4 == 2:
        for j in range(b):
            for i in range(a):
                ii = (-i) % a
                jj = (-j) % b
                newState.append(state[a*jj+ii])
    else:
        for j in range(b):
            for i in range(a):
                ii = (-j) % a
                jj = i % b
                newState.append(state[a*jj+ii])

    return np.array(newState)

def rotCentSym(state,n): # rotate about the center of the primitive cell [0-1]x[0-1] by n\pi/2 radians CCW 
    newState = []
    if n % 4 == 0:
        return state
    elif n % 4 == 1:
        for j in range(b):
            for i in range(a):
                ii = (j-0.5) % a 
                jj = (-i+0.5) % b
                ii = (ii+0.5) % a
                jj = (jj+0.5) % b
                newState.append(state[int(a*jj+ii)])
    elif n % 4 == 2:
        for j in range(b):
            for i in range(a):
                ii = (-i+0.5) % a
                jj = (-j+0.5) % b
                ii = (ii+0.5) % a
                jj = (jj+0.5) % b
                newState.append(state[int(a*jj+ii)])
    else:
        for j in range(b):
            for i in range(a):
                ii = (-j+0.5) % a 
                jj = (i-0.5) % b
                ii = (ii+0.5) % a
                jj = (jj+0.5) % b
                newState.append(state[int(a*jj+ii)])
    return np.array(newState)

def rotHorEdgeSym(state): # rotate about the center of the edge connecting (0,0) with (1,0) by \pi radians

    newState = []
    for j in range(b):
        for i in range(a):
            ii = (-i+0.5) % a
            jj = (-j) % b
            ii = (ii+0.5) % a
            newState.append(state[int(a*jj+ii)])
    return np.array(newState)

def rotVertEdgeSym(state): # rotate about the center of the edge connecting (0,0) with (0,1) by \pi radians
    newState = []
    for j in range(b):
        for i in range(a):
            jj = (-j+0.5) % b
            ii = (-i) % a
            jj = (jj+0.5) % b
            newState.append(state[int(a*jj+ii)])
    return np.array(newState)

def refVertSym(state): # reflect about the line x = 0
    newState = []
    for j in range(b):
        for i in range(a):
            ii = (-i) % a
            jj = j
            newState.append(state[int(a*jj+ii)])
    return np.array(newState)

def refHorSym(state): # reflect about the line y = 0
    newState = []
    for j in range(b):
        for i in range(a):
            ii = i
            jj = (-j) % b
            newState.append(state[int(a*jj+ii)])
    return np.array(newState)

def refVertCentSym(state): # reflect about the line x = 0.5
    newState = []
    for j in range(b):
        for i in range(a):
            ii = (-i+0.5) % a
            jj = j
            ii = (ii+0.5) % a
            newState.append(state[int(a*jj+ii)])
    return np.array(newState)

def refHorCentSym(state): # reflect about the line y = 0.5
    newState = []
    for j in range(b):
        for i in range(a):
            ii = i
            jj = (-j+0.5) % b
            jj = (jj+0.5) % b
            newState.append(state[int(a*jj+ii)])
    return np.array(newState)

def refDiagSym(state): # reflect about the line y = x
    newState = []
    for j in range(b):
        for i in range(a):
            ii = j
            jj = i
            newState.append(state[int(a*jj+ii)])
    return np.array(newState)

def refOffDiagSym(state): # reflect about the line y = -x
    newState = []
    for j in range(b):
        for i in range(a):
            ii = (-j) % a
            jj = (-i) % b
            newState.append(state[int(a*jj+ii)])
    return np.array(newState)

identity = np.arange(n)
isSquare = (a == b)

gen = []

gen.append(transSym(identity,1,0))
gen.append(transSym(identity,0,1))

if isSquare:
    gen.append(rotSiteSym(identity,1))
    gen.append(rotCentSym(identity,1))
    gen.append(rotHorEdgeSym(identity))
    
    gen.append(refVertSym(identity))
    gen.append(refVertCentSym(identity))
    gen.append(refDiagSym(identity))
else:
    gen.append(rotSiteSym(identity,2))
    gen.append(rotCentSym(identity,2))
    gen.append(rotHorEdgeSym(identity))
    gen.append(rotVertEdgeSym(identity))

    gen.append(refVertSym(identity))
    gen.append(refVertCentSym(identity))
    gen.append(refHorSym(identity))
    gen.append(refHorCentSym(identity))

G = PermutationGroup(gen)

orbits = []

def isVisited(index):
    for orbit in orbits:
        if index in orbit:
            return True
    return False

ptr = 0

while ptr < N:
    orbit = []
    orbit.append(ptr)
    for g in G.generate():
        newState = np.array([stateTable[ptr][g(i)] for i in range(n)])
        orbit.append(searchState(newState))
        if n % 2 == 0: orbit.append(searchState(spinFlipSym(newState)))
    orbit = list(set(orbit))
    orbits.append(orbit)
    while isVisited(ptr):
        ptr += 1

orbitLengths = [len(orbits[i]) for i in range(len(orbits))]

sns.histplot(orbitLengths,bins=50)
plt.show()
NN = len(orbits)
print('number of orbits:'+ NN)

def computeHamEntry(J1,J2,i,j):
    sum = 0.0
    if (i == j):
        for k in range(0,n-1):
            for l in range(k+1,n):
                if (N1[k][l] != 0):
                    sum += J1*stateTable[i][k]*stateTable[i][l]
                if (N2[k][l] != 0):
                    sum += J2*stateTable[i][k]*stateTable[i][l]          
    else:
        tempState = np.multiply(stateTable[i],stateTable[j])
        if (np.count_nonzero(tempState == -1) == 2):
            indices = np.where(tempState == -1)
            e,f = indices[0][0],indices[0][1]
            if (N1[e][f] != 0):
                sum += 2*J1
            if (N2[e][f] != 0):
                sum += 2*J2
    
    return sum

def computeRedHam(J1,J2):
    row = []
    col = []
    data = []

    for i in range(0,NN):
        for j in range(i,NN):
            if (i == j):
                row.append(i)
                col.append(j)
                data.append(computeHamEntry(J1,J2,orbits[i][0],orbits[j][0]))
            else:
                sum = 0
                if len(orbits[i]) <= len(orbits[j]):
                    for k in orbits[i]:
                        sum += computeHamEntry(J1,J2,k,orbits[j][0])
                    sum *= len(orbits[j])
                else:
                    for k in orbits[j]:
                        sum += computeHamEntry(J1,J2,orbits[i][0],k)
                    sum *= len(orbits[i])
                sum /= np.sqrt(len(orbits[i])*len(orbits[j]))

                row.append(i)
                row.append(j)
                col.append(j)
                col.append(i)
                data.append(sum)
                data.append(sum)
    
    H = sparse.coo_array((data, (row, col)), shape=(NN,NN), dtype=np.float32)
    return H

H = computeRedHam(1,0)
end = time.time()

print(orbits)
print(H.toarray())
numEnergies = 1
energies, states = sparse.linalg.eigs(H.asfptype(), k=numEnergies, which='SR')
energies = np.real(energies)
states = np.real(states)
print('ground state energy:' + energies[0])

def toFullBasis(reducedBasisState):
    ret = np.zeros(N)
    for i in range(len(orbits)):
        for j in range(len(orbits[i])):
            ret[orbits[i][j]] = reducedBasisState[i]/np.sqrt(len(orbits[i]))

    return ret

fullState = toFullBasis(states.T[0])
print('ground state:' + fullState)
np.savetxt(PATH + '/data/6x6grd.txt', fullState, delimiter=',', fmt='%d')


print('total time to run:' + str(end-start))