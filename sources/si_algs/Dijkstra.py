from mpi4py import MPI
import numpy as np
import time
import random
import math
# import importlib

class Dijkstra:
    para_count = 0
    # para_names = ['first', 'second', 'third']
    # para_types = [0, 2, 1]
    # para_range_starts = [1, 0, 20]
    # para_range_ends = [30, 0, 40]
    # para_def_values = [10, 1, 35]
    # para_cat_values = [[], ['basic', 'advanced', 'expert'], []]

def run(graph, source, dest):
    N = len(graph)
    Q = list(range(N))
    dist = [9999999] * N
    prev = [-1] * N    
        
    dist[source] = 0    
    
    while Q:
        minv = 9999999
        u = Q[0]
        for i in Q:
            if dist[i] < minv:
                minv = dist[i]
                u = i
        if u == dest:
            x = dest
            while prev[x] != source and prev[x] != -1:
                x = prev[x]
            if prev[x] == -1:
                return -1
            else:
                return x
        Q.remove(u)
        
        for v in range(len(graph[u])):
            alt = 9999999
            if graph[u][v] > 0 and v in Q:
                alt = dist[u] + graph[u][v]
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
    
    # return prev[dest]
    # path = []
    # a = dest
    # while a != source:
    #     path.append(prev[a])
    #     a = prev[a]
    # return path[-1]

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
    wfp100 = glob_paras[WAKEFULPROB]
    dataprob = glob_paras[DATAORIGINPROB]
    sensorCount = glob_paras[SENSORCOUNT]
    ad0 = glob_paras[SENSD0]
    oneMeter = ad0 / METERD0
    bLoss = glob_paras[BPATHLOSS]
    alfa = glob_paras[ALPHA]
    oneMinAlfa = 1 - alfa
    # get alg spec paras
    alg_spec_paras = None
    alg_spec_paras = comm.bcast(alg_spec_paras, root=0)
    
    # discover neighbors--------------------------------------------------------------------------
    # neighbors_ids = [[-1] * sensorCount] * sensorCount
    neighbors_dists = [[-1] * (sensorCount + 1) for _ in range(sensorCount + 1)]
    maxtav = glob_paras[HATOTAV] / oneMeter
    maxtav2 = glob_paras[HATOTAV] * glob_paras[HATOTAV]
    for i in range(sensorCount):
        for j in range(i):
            x = koord_array[j][0] - koord_array[i][0]
            y = koord_array[j][1] - koord_array[i][1]
            tav = x * x + y * y
            if tav <= maxtav2:
                neighbors_dists[i][j] = math.sqrt(tav) / oneMeter
                neighbors_dists[j][i] = neighbors_dists[i][j]
    # distance from the sink
    offs = 9
    distsFromSink = [0] * sensorCount
    for i in range(sensorCount):
        xdel = koord_array[i][0] - offs; ydel = koord_array[i][1] - offs
        distsFromSink[i] = math.sqrt(xdel * xdel + ydel * ydel) / oneMeter
    # Is sink in comm. range?
    isSinkAvailArr = [0] * sensorCount
    for i in range(sensorCount):
        isSinkAvailArr[i] = (distsFromSink[i] <= maxtav)
    # complete dist array with distances from sink
    for i in range(sensorCount):
        neighbors_dists[i][sensorCount] = (distsFromSink[i] if isSinkAvailArr[i] else -1)
    for i in range(sensorCount):
        neighbors_dists[sensorCount][i] = (distsFromSink[i] if isSinkAvailArr[i] else -1)
    # sort the distance from sink tomb, and create index array
    idxSortedDistSink = list(range(sensorCount))
    for i in range(sensorCount):
        maxIdx = i
        for j in range(i + 1, sensorCount):
            if distsFromSink[j] > distsFromSink[maxIdx]:
                maxIdx = j
        if maxIdx != i:
            distsFromSink[i], distsFromSink[maxIdx] = distsFromSink[maxIdx], distsFromSink[i]
            idxSortedDistSink[i], idxSortedDistSink[maxIdx] = idxSortedDistSink[maxIdx], idxSortedDistSink[i]
    # print("idxSortedDistSink", idxSortedDistSink)
    
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
    # print("proc", rank, sharedMem[0], sharedMem[1])
    # take out datas from shared memory
    # awake: 1, not awake: 0
    wakeFulArray = sharedMem[simGlobParaCount:sensorCount + simGlobParaCount]
    # for i in range(len(wakeFulArray)):
    for i in range(sensorCount):
        wakeFulArray[i] = (1 if random.random() <= wfp100 else 0)
    battCapArray = sharedMem[sensorCount + simGlobParaCount:2 * sensorCount + simGlobParaCount]
    initBattCap = battCapArray.copy()
    sensorsData = sharedMem[2 * sensorCount + simGlobParaCount:].reshape(itemPerProc, sensorCount)
    # tolas = 2 * sensorCount + 4
    # sensorsData = [0] * sensorCount
    # for i in range(sensorCount):
    #     sensorsData[i] = sharedMem[tolas + i * itemPerProc:tolas + (i + 1) * itemPerProc]
    # fill parent shared mem with this process data
    # sharedMem[:5] = np.arange(5)
    
    # init sim. params------------------------------------------------------------------------------
    # Efs = 10 pJ / bit / m^2
    Efs = 0.00000000001
    # costs
    oneBitRecCost = 0.00000005
    # E0 = 2.5, max sug en = Eoutmax = E0 * 10^-5 * 25 * d0^2
    framelen = glob_paras[FRAMELEN] * 8
    # oneBcastCost = 2.5 * 0.00001 * 25 * framelen
    #            = Efs *   (5 * d0)^2   * framelen
    oneBcastCost = Efs * 25 * METERD0**2 * framelen
    recCost = oneBitRecCost * framelen
    readyStateCost = 80 * oneBitRecCost
    # modositast igenyelhet a 0.01
    rc_tenyezo = 0.01 * oneBitRecCost
    routalgCost = rc_tenyezo * sensorCount**2 * math.log(sensorCount, 10)
    enoughEnergy = 2 * recCost
    # others
    lenInputQueues = [0] * sensorCount
    inputQueues = [[] for i in range(sensorCount)]
    sourceQueues = [[] for i in range(sensorCount)]
    sinkHopArray = []
    sinkSourceArray = []
    areaAtlo = math.sqrt(5) * METERD0 * glob_paras[AREARECTH]
    enhBattCap = glob_paras[BATTEN]
    EoutNev = METERD0**(bLoss - 2)
    prevSped = -1
    roundTime = 1 / 25
    # kommN = int(sensorCount * glob_paras[ITERFORMEAS] / 100)
    kommN = glob_paras[ITERFORMEAS]
    countkommn = [0] * sensorCount
    for i in range(sensorCount):
        countkommn[i] = int(random.random() * kommN)
    isDataOrig = [0] * sensorCount
    for i in range(sensorCount):
        isDataOrig[i] = (random.random() <= dataprob)
    endSign = False
    maxGapTime = 0.2 * 20 / sensorCount
    gapTime = 1
    maxPairCount = int(commPairItemCount / 2)
    
    # calculate sending costs
    sendCosts = [[0] * (sensorCount + 1) for _ in range(sensorCount)]
    for i in range(sensorCount):
        for j in range(i):
            R = neighbors_dists[i][j]
            if R != -1:
                sendCosts[i][j] = (Efs * R**2 * framelen if R <= METERD0 else ((Efs * R**bLoss) / EoutNev) * framelen)
                sendCosts[j][i] = sendCosts[i][j]
    for j in range(sensorCount):
        R = neighbors_dists[j][sensorCount]
        if R != -1:
            sendCosts[j][sensorCount] = (Efs * R**2 * framelen if R <= METERD0 else ((Efs * R**bLoss) / EoutNev) * framelen)
    # set half metrics
    for uu in range(sensorCount + 1):
        for jj in range(uu):
            if neighbors_dists[uu][jj] != -1:
                asd = (neighbors_dists[uu][jj] / areaAtlo) * alfa
                neighbors_dists[uu][jj] = asd
                neighbors_dists[jj][uu] = asd
    # set metrics to sink
    # for jj in range(sensorCount):
    #     if neighbors_dists[jj][sensorCount] != -1:
    #         neighbors_dists[jj][sensorCount] = (neighbors_dists[jj][sensorCount] / areaAtlo) * alfa
    # set enhanced batt. * (1 - alpha)
    enhBCmultOneMinAlfa = enhBattCap * oneMinAlfa
    rFD = 10
    multFlood = 0
    for i in range(sensorCount):
        for j in range(sensorCount):
            if neighbors_dists[i][j] != -1:
                multFlood += 1
    multFlood /= sensorCount
    multFlood = math.ceil(sensorCount / multFlood)
    # initBcastCost: where the sensors, flood the distances
    initBcastCost = sensorCount * wfp100 * recCost + oneBcastCost * multFlood
    for i in range(sensorCount):
        battCapArray[i] -= initBcastCost
    
    recIdxArr = [0] * sensorCount
    tmpDists = [neighbors_dists[gg].copy() for gg in range(sensorCount + 1)]
    for uu in range(sensorCount):
        for jj in range(uu):
            if tmpDists[uu][jj] != -1:
                if battCapArray[jj] > enoughEnergy and battCapArray[uu] > enoughEnergy and wakeFulArray[uu] and wakeFulArray[jj]:
                    tmpDists[uu][jj] += enhBCmultOneMinAlfa / battCapArray[jj]
                    tmpDists[jj][uu] += enhBCmultOneMinAlfa / battCapArray[uu]
                else:
                    tmpDists[uu][jj] = -1
                    tmpDists[jj][uu] = -1
    for jj in range(sensorCount):
        if tmpDists[sensorCount][jj] != -1:
            if not (battCapArray[jj] > enoughEnergy and wakeFulArray[jj]):
                tmpDists[sensorCount][jj] = -1
                tmpDists[jj][sensorCount] = -1
    # calculate who to send, source: sensor, dest: sink
    # compute flooding cost (get chargelevels), self does not matter
    liveNoSleepNodes = -1
    for ll in range(sensorCount):
        # if battCapArray[ll] > 0 and wakeFulArray[ll]:
        if wakeFulArray[ll]:
            liveNoSleepNodes += 1
    floodCost = liveNoSleepNodes * recCost + oneBcastCost * multFlood
    for i in range(sensorCount):
        recIdxArr[i] = run(tmpDists, i, sensorCount)
        sensorsData[8][i] += floodCost
        # subtract floodCost + routingAlgCost
        battCapArray[i] -= floodCost + routalgCost
    
    # print('multFlood:', multFlood)
    # print("koord_array:", koord_array)
    # print("glob_paras:", glob_paras)
    # print("wfp100:", wfp100)
    # print("dataprob:", dataprob)
    # print("sensorCount:", sensorCount)
    # print("ad0:", ad0)
    # print("oneMeter:", oneMeter)
    # print("bLoss:", bLoss)
    # print("alfa:", alfa)
    # print("oneMinAlfa:", oneMinAlfa)
    # print("alg_spec_paras:", alg_spec_paras)
    # print("neighbors_dists:", neighbors_dists)
    # print("maxtav:", maxtav)
    # print("maxtav2:", maxtav2)
    # print("distsFromSink:", distsFromSink)
    # print("isSinkAvailArr:", isSinkAvailArr)
    # print("idxSortedDistSink:", idxSortedDistSink)
    # print("wakeFulArray:", wakeFulArray)
    # print("battCapArray:", battCapArray)
    # print("sensor data", sensorsData)
    # print("framelen:", framelen)
    # print("oneBcastCost:", oneBcastCost)
    # print("recCost:", recCost)
    # print("readyStateCost:", readyStateCost)
    # print("routalgCost:", routalgCost)
    # print("initBcastCost:", initBcastCost)
    # print("enoughEnergy:", enoughEnergy)
    # print("areaAtlo:", areaAtlo)
    # print("enhBattCap:", enhBattCap)
    # print("EoutNev:", EoutNev)
    # print("kommN:", kommN)
    # print("countkommn:", countkommn)
    # print("isDataOrig:", isDataOrig)
    # print("enhBCmultOneMinAlfa:", enhBCmultOneMinAlfa)
    # print("sendCosts:", sendCosts)
    # sharedMem[2] = 1
    
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
                # round start compute
                if wakeFulArray[dex] and countkommn[dex] == 0:
                    # flood cost, dijstra, send cost-------------------------------------------------------
                    # compute flooding cost (get chargelevels), self does not matter
                    liveNoSleepNodes = -1
                    for ll in range(sensorCount):
                        # if battCapArray[ll] > 0 and wakeFulArray[ll]:
                        if wakeFulArray[ll]:
                            liveNoSleepNodes += 1
                    floodCost = liveNoSleepNodes * recCost + oneBcastCost * multFlood
                    sensorsData[8][dex] += floodCost
                    # subtract floodCost + routingAlgCost
                    battCapArray[dex] -= floodCost + routalgCost
                    if battCapArray[dex] <= 0:
                        battCapArray[dex] = -1
                        wakeFulArray[dex] = 0
                        continue
                    # modify the dist array with the actual charge levels
                    tmpDists = [neighbors_dists[gg].copy() for gg in range(sensorCount + 1)]
                    for uu in range(sensorCount):
                        for jj in range(uu):
                            if tmpDists[uu][jj] != -1:
                                if battCapArray[jj] > enoughEnergy and battCapArray[uu] > enoughEnergy and wakeFulArray[uu] and wakeFulArray[jj]:
                                    tmpDists[uu][jj] += enhBCmultOneMinAlfa / battCapArray[jj]
                                    tmpDists[jj][uu] += enhBCmultOneMinAlfa / battCapArray[uu]
                                else:
                                    tmpDists[uu][jj] = -1
                                    tmpDists[jj][uu] = -1
                    for jj in range(sensorCount):
                        if tmpDists[sensorCount][jj] != -1:
                            if not (battCapArray[jj] > enoughEnergy and wakeFulArray[jj]):
                                tmpDists[sensorCount][jj] = -1
                                tmpDists[jj][sensorCount] = -1
                    # calculate who to send, source: sensor, dest: sink
                    recIdxArr[dex] = run(tmpDists, dex, sensorCount)
                # ha ebren van, es van sajat adatot kuldeni es/vagy van mit tovabbitani
                if wakeFulArray[dex] and (isDataOrig[dex] or lenInputQueues[dex] > 0):
                    neighbToSink = recIdxArr[dex]
                    # print("dex", dex, "nts", neighbToSink, flush=True)
                    # if no available neighbor then continue
                    if neighbToSink == -1:
                        # print("rout alg failure", flush=True)
                        # print(wakeFulArray, flush=True)
                        sensorsData[10][dex] += 1
                        countkommn[dex] += 1
                        continue
                    # get sending cost
                    actSendCost = sendCosts[dex][neighbToSink]
                    #-----------------------------------------------------------------------------------------------
                    nemVoltPair = True
                    # data origin prob
                    if isDataOrig[dex]:
                        # increase the done meas. stat.
                        sensorsData[7][dex] += 1
                        # false till next round
                        isDataOrig[dex] = False
                        # subtract sending cost
                        battCapArray[dex] -= actSendCost
                        if battCapArray[dex] < 0:
                            battCapArray[dex] = -1
                            wakeFulArray[dex] = 0
                            # sajat keretvesztes
                            sensorsData[4][dex] += 1
                            continue
                        # inc. total energy
                        sensorsData[8][dex] += actSendCost
                        # if send to sink
                        nemVoltPair = False
                        # ha a komm parok szama = max
                        if sharedMem[4] == maxPairCount:
                            sharedMem[4] = 0
                        # komm. pair mentese
                        commPairData[int(sharedMem[4] * 2)] = dex
                        commPairData[int(sharedMem[4] * 2 + 1)] = neighbToSink
                        sharedMem[4] += 1
                        # send to sink
                        if neighbToSink == sensorCount:
                            # sinkhez erkezo keretek szama
                            sharedMem[3] += 1
                            # other's data to sink stat
                            sensorsData[0][dex] += 1
                            sinkHopArray.append(1)
                            sinkSourceArray.append(dex)
                            # sinkhez erkezett keretek dex forrasbol
                            sensorsData[9][dex] += 1
                            sensorsData[11][dex] += 1
                        else:
                            # subtract receiving cost at the receiver
                            sensorsData[8][neighbToSink] += recCost
                            battCapArray[neighbToSink] -= recCost
                            # notify neighbor about sending
                            lenInputQueues[neighbToSink] += 1
                            inputQueues[neighbToSink].append(1)
                            sourceQueues[neighbToSink].append(dex)
                            # send to szomszed stat
                            sensorsData[1][dex] += 1
                    # start forwarding state
                    # process the input queue
                    if lenInputQueues[dex] > 0:
                        iqdex = lenInputQueues[dex]
                        # subtract sending cost
                        dsa = actSendCost * iqdex
                        if battCapArray[dex] < dsa:
                            # mennyit tud elkuldeni
                            iqdex = int(battCapArray[dex] / actSendCost)
                            # inc total energy
                            sensorsData[8][dex] += iqdex * actSendCost
                            # lost frames
                            sensorsData[5][dex] += lenInputQueues[dex] - iqdex
                            battCapArray[dex] = -1
                            wakeFulArray[dex] = 0
                        else:
                            battCapArray[dex] -= dsa
                            # inc total energy
                            sensorsData[8][dex] += dsa
                        # increment the first iqdex db hop counts in the input queue
                        for i in range(iqdex):
                            inputQueues[dex][i] += 1
                        # send to sink or neighbor
                        if neighbToSink == sensorCount:
                            # sinkhez erkezo keretek szama
                            sharedMem[3] += iqdex
                            # masik adat to sink
                            sensorsData[2][dex] += iqdex
                            # ha a fogadott keretek szama > 0 es nem volt sikeres sajat adat kuldes
                            if iqdex > 0 and nemVoltPair:
                                # komm pairs kezeles
                                if sharedMem[4] == maxPairCount:
                                    sharedMem[4] = 0
                                commPairData[int(sharedMem[4] * 2)] = dex
                                commPairData[int(sharedMem[4] * 2 + 1)] = neighbToSink
                                sharedMem[4] += 1
                            sinkHopArray.extend(inputQueues[dex][:iqdex])
                            sinkSourceArray.extend(sourceQueues[dex][:iqdex])
                            # sinkhez erkezett keretek az adott forrasbol
                            zmt = sourceQueues[dex][:iqdex]
                            for kk in range(iqdex):
                                sensorsData[9][zmt[kk]] += 1
                                sensorsData[11][zmt[kk]] += inputQueues[dex][kk]
                        else:
                            # subtract receiving cost at the receiver
                            dsa = recCost * iqdex
                            if battCapArray[neighbToSink] < dsa:
                                sikeres = int(battCapArray[neighbToSink] / recCost)
                                if sikeres < 0:
                                    sikeres = 0
                                battCapArray[neighbToSink] = -1
                                wakeFulArray[neighbToSink] = 0
                                # send to szomszed stat
                                sensorsData[3][dex] += sikeres
                                # komm pairs kezeles
                                if sikeres > 0 and nemVoltPair:
                                    if sharedMem[4] == maxPairCount:
                                        sharedMem[4] = 0
                                    commPairData[int(sharedMem[4] * 2)] = dex
                                    commPairData[int(sharedMem[4] * 2 + 1)] = neighbToSink
                                    sharedMem[4] += 1
                                # keretvesztes - mas adatat nem sikerult kuldeni
                                sensorsData[5][neighbToSink] += iqdex - sikeres
                                # total energy for all comm.
                                sensorsData[8][neighbToSink] += sikeres * recCost
                            else:
                                battCapArray[neighbToSink] -= dsa
                                # total energy for all comm.
                                sensorsData[8][neighbToSink] += dsa
                                # notify neighbor about sending
                                lenInputQueues[neighbToSink] += iqdex
                                inputQueues[neighbToSink].extend(inputQueues[dex][:iqdex])
                                sourceQueues[neighbToSink].extend(sourceQueues[dex][:iqdex])
                                # send to szomszed stat
                                sensorsData[3][dex] += iqdex
                                # komm pairs kezeles
                                if iqdex > 0 and nemVoltPair:
                                    if sharedMem[4] == maxPairCount:
                                        sharedMem[4] = 0
                                    commPairData[int(sharedMem[4] * 2)] = dex
                                    commPairData[int(sharedMem[4] * 2 + 1)] = neighbToSink
                                    sharedMem[4] += 1
                        # reset the inputQueue
                        lenInputQueues[dex] = 0
                        inputQueues[dex] = []
                        sourceQueues[dex] = []
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
    
    print("vege", flush=True)
    universe.send(sinkHopArray, dest=parRank, tag=1)
    universe.send(sinkSourceArray, dest=parRank, tag=1)
    universe.Disconnect()
    comm.Disconnect()
