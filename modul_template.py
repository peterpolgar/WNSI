from mpi4py import MPI
import numpy as np
import time
import random
import math
# import importlib

class A_routing_alg:
    para_count = 0
    # para_names = ['first', 'second', 'third']
    # para_types = [0, 2, 1]
    # para_range_starts = [1, 0, 20]
    # para_range_ends = [30, 0, 40]
    # para_def_values = [10, 1, 35]
    # para_cat_values = [[], ['basic', 'advanced', 'expert'], []]

# You can define any number of functions and classes with any parameters and classes you want, for example:
def fn_for_routing_1(graph, source, dest):
    pass

if __name__ == "__main__":
    random.seed()
    # create const indexes
    ALPHA = 0
    BPATHLOSS = 1
    DATAORIGINPROB = 2
    FRAMELEN = 3
    WAKEFULPROB = 4
    HATOTAV = 5
    SENSD0 = 6
    SENSORCOUNT = 7
    ALIVETHRES = 8
    ITERFORMEAS = 9
    AREARECTH = 10
    BATTEN = 11
    
    METERD0 = 87.5
    
    # creat MPI comm
    comm = MPI.Comm.Get_parent()
    # a sensor koordinatak fogadasa
    koord_array = None
    koord_array = comm.bcast(koord_array, root=0)
    # get global paras
    glob_paras = None
    glob_paras = comm.bcast(glob_paras, root=0)
    # get alg spec paras
    alg_spec_paras = None
    alg_spec_paras = comm.bcast(alg_spec_paras, root=0)
    
    # get shared memory--------------------------------------------------------------------------------
    universe = MPI.Intercomm.Merge(comm)
    rankUni = universe.Get_rank()
    # get parent uni rank
    parRank = 0
    parRank = comm.bcast(parRank, root=0)
    # print("parRank", parRank)
    
    # create own shared memory
    itemsize = MPI.DOUBLE.Get_size()
    commPairItemCount = 10000
    contShCount = 8
    oneCountSize = 4000
    shitemcount = (commPairItemCount + contShCount * oneCountSize)
    nbytes = shitemcount * itemsize
    win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=universe)
    myShData = np.ndarray(buffer=win.tomemory(), dtype='d', shape=(shitemcount))
    myShData.fill(0)
    commPairData = myShData[:commPairItemCount]
    contShMem = [0] * contShCount
    for i in range(contShCount):
        contShMem[i] = myShData[commPairItemCount + i * oneCountSize:commPairItemCount + (i + 1) * oneCountSize]
    # wait until parent create and fill shared mem
    universe.Barrier()
    # initialization with parent ended----------------------------------
    
    #get access to parent shared mem
    buf, itemsize = win.Shared_query(parRank)
    assert itemsize == MPI.DOUBLE.Get_size()
    # create array from the pointer
    itemPerProc = 12
    simGlobParaCount = 5
    size = sensorCount * (itemPerProc + 2) + simGlobParaCount
    sharedMem = np.ndarray(buffer=buf, dtype='d', shape=(size))
    wakeFulArray = sharedMem[simGlobParaCount:sensorCount + simGlobParaCount]
    for i in range(sensorCount):
        wakeFulArray[i] = (1 if random.random() <= wfp100 else 0)
    battCapArray = sharedMem[sensorCount + simGlobParaCount:2 * sensorCount + simGlobParaCount]
    sensorsData = sharedMem[2 * sensorCount + simGlobParaCount:].reshape(itemPerProc, sensorCount)
    
    # init sim. params------------------------------------------------------------------------------
    # code...
    
    # do sim.
    while True:
        for t in range(sensorCount):
            # if szim should run
            if sharedMem[0] == 0:
                dex = idxSortedDistSink[t]
                # if this sensor discharged, then cont.
                if battCapArray[dex] == -1:
                    # examine if sim should end, because battery cap. reach threshold
                    asum = 0
                    acount = 0
                    for i in range(sensorCount):
                        if wakeFulArray[i]:
                            asum += battCapArray[i]
                            acount += 1
                    if acount == 0 or asum / acount < glob_paras[ALIVETHRES]:
                        endSign = True
                        sharedMem[2] = 1
                        break
                    continue
                # if one round expired
                if countkommn[dex] == kommN:
                    # set count. sh. mem
                    if sensorsData[6][dex] % rFD == 0:
                        za = int(sensorsData[6][dex] / rFD) - 1
                        if za != -1:
                            contShMem[0][za] = sharedMem[3]
                            if sensorsData[9][dex] != 0:
                                contShMem[1][za] += sensorsData[11][dex] / sensorsData[9][dex]
                            nev = sensorsData[0][dex] + sensorsData[1][dex] + sensorsData[2][dex] + sensorsData[3][dex] + sensorsData[4][dex] + sensorsData[5][dex] + sensorsData[10][dex]
                            if nev != 0:
                                contShMem[2][za] += (sensorsData[4][dex] + sensorsData[5][dex] + sensorsData[10][dex]) / nev
                            contShMem[3][za] += battCapArray[dex] / initBattCap[dex]
                            contShMem[4][za] += sensorsData[2][dex] + sensorsData[3][dex]
                            contShMem[5][za] += sensorsData[8][dex] / initBattCap[dex]
                            contShMem[6][za] += sensorsData[10][dex]
                            contShMem[7][za] += 1
                    # examine if sim should end, because battery cap. reach threshold
                    asum = 0
                    acount = 0
                    for i in range(sensorCount):
                        if wakeFulArray[i]:
                            asum += battCapArray[i]
                            acount += 1
                    if acount == 0 or asum / acount < glob_paras[ALIVETHRES]:
                        endSign = True
                        sharedMem[2] = 1
                        break
                    # set round count
                    sensorsData[6][dex] += 1
                    # set count of the next round to 0
                    countkommn[dex] = 0
                    # decide awake or not
                    isAwake = (1 if random.random() <= wfp100 else 0)
                    if wakeFulArray[dex] != isAwake:
                        wakeFulArray[dex] = isAwake
                    # subtract ready state cost
                    if isAwake:
                        battCapArray[dex] -= readyStateCost
                        if battCapArray[dex] <= 0:
                            battCapArray[dex] = -1
                            wakeFulArray[dex] = 0
                            continue
                        # decide data origin
                        isDataOrig[dex] = (random.random() <= dataprob)
                # do one round
                # code...
                countkommn[dex] += 1
                # set gap time to slow down the execution (speed)
                if sharedMem[1] != prevSped:
                    prevSped = sharedMem[1]
                    gapTime = (100 - prevSped) / 100 * maxGapTime
                time.sleep(gapTime)
            # if szim stopped
            elif sharedMem[0] == 1:
                while sharedMem[0] == 1:
                    time.sleep(roundTime)
            # if szim forced to finish
            else:
                endSign = True
                break
        if endSign:
            break
    
    universe.send(sinkHopArray, dest=parRank, tag=1)
    universe.send(sinkSourceArray, dest=parRank, tag=1)
    universe.Disconnect()
    comm.Disconnect()
