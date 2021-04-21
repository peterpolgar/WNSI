from mpi4py import MPI
import numpy as np
import rc_icons
import mycallout
import sys
import math
import random
import poisson_disk
import importlib
import pkgutil
import pickle
from PySide2 import QtCore
from PySide2.QtGui import QIcon, QPalette, QColor, QPainter, QFont, QLinearGradient, QPen, QBrush, QPolygon, QPixmap, QTextDocument
from PySide2.QtCore import Qt, Slot, QTimer, QPoint, QPointF, QFile, QIODevice
from PySide2.QtWidgets import (QApplication, QWidget, QListWidget, QListWidgetItem,
                               QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QSlider,
                               QStyle, QSizePolicy, QLayout, QBoxLayout, QComboBox, QGridLayout,
                               QScrollArea, QStackedLayout, QButtonGroup, QStyleOptionSlider,
                               QAbstractSlider, QCheckBox, QRadioButton, QFileDialog)
from PySide2.QtCharts import QtCharts


class Sim_run:
    def __init__(self, moduleid, isGroupRun, runid, par):
        self.moduleid = moduleid
        self.isGroupRun = isGroupRun
        self.runid = runid
        self.par = par
        self.glob_paras = []
        self.spec_paras = []
        
        self.sens_datas = []
        self.grafSeries = [QtCharts.QLineSeries() for i in range(7)]
        self.universe = None
        self.comm = None
        self.procData = None
        self.wakefulArray = None
        self.battCapArray = None
        self.isEnhancedSensors = None
        self.initBattCapArray = None
        self.commPairMem = None
        self.grafShMem = None
        self.numProc = 5
        # number of items of data of one proc
        self.itemPerProc = 12
        # self.maxRound = 5
        self.timer = QTimer()
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.drawData)
        self.remainderTime = 0
        self.drawInterval = 210
        self.basicen = 0
        self.aliveThres = 0
        self.statMax = 0
        self.statMin = 0
        self.offsetCommPair = 0
        self.simGlobParaCount = 5
        self.locOfSens = None
        self.sensSug = 0
        self.hatotav = 0
        self.cpSize = 0
    
    def begRun(self):
        self.hatotav = self.par.hatotav
        self.locOfSens = self.par.sensor_locs.copy()
        self.sensSug = self.par.sim_widget.sens_sug
        # set graph variables
        for i in range(7):
            gpen = self.grafSeries[i].pen()
            gpen.setWidth(5)
            self.grafSeries[i].setPen(gpen)
            self.grafSeries[i].setName("Live run")
        self.par.liveOffset = 0
        if self.par.autoGrDr.isChecked() and self.par.drawinto_bg.checkedId() == 0:
            self.par.deSelectAll()
        # set number of processes
        self.numProc = self.glob_paras[self.par.NOSENSORS]
        # set aliveThres
        self.aliveThres = (self.glob_paras[self.par.ALIBATTHRES] / 100) * self.glob_paras[self.par.BATTEN]
        #create and start sensor processes--------------------------------------------------------------------
        self.comm = MPI.COMM_SELF.Spawn(sys.executable,
                                   args=["./si_algs/" + self.par.routing_alg_names[self.moduleid] + ".py"],
                                   maxprocs=1)
        # self.comm = MPI.COMM_SELF.Spawn(sys.executable,
        #                            args=["./si_algs/" + par.si_select_list.currentItem().text() + ".py"],
        #                            maxprocs=1)
        #-----------------------------------------------------
        # a sensor koordinatak szetkuldese
        self.comm.bcast(self.par.sensor_locs, root=MPI.ROOT)
        # a parameterek szetkuldese
        self.comm.bcast([self.glob_paras[self.par.ALPHA], self.glob_paras[self.par.BPATHLOSS], self.glob_paras[self.par.DATAORIGINPROB], self.glob_paras[self.par.FRAMELEN], self.glob_paras[self.par.WAKEFULPROB], self.par.hatotav, self.par.sensd0, self.numProc, self.aliveThres, self.glob_paras[self.par.ITERFORMEAS], self.glob_paras[self.par.AREARECTH], self.glob_paras[self.par.BATTEN] * (1 + self.glob_paras[self.par.EXTRAENERGY] / 100)], root=MPI.ROOT)
        self.comm.bcast(self.spec_paras, root=MPI.ROOT)
        ## create shared memory
        # merge
        self.universe = MPI.Intercomm.Merge(self.comm)
        # get this (parent) rank
        rank = self.universe.Get_rank()
        # send parent rank
        self.comm.bcast(rank, root=MPI.ROOT)
        # all memory size
        size = self.numProc * (self.itemPerProc + 2) + self.simGlobParaCount
        itemsize = MPI.DOUBLE.Get_size()
        nbytes = size * itemsize
        # allocate memory
        win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=self.universe)
        # pointer to memory
        buf = win.tomemory()
        # assert itemsize == MPI.DOUBLE.Get_size()
        # reformat the buf pointer
        self.sens_datas = np.ndarray(buffer=buf, dtype='d', shape=(size))
        
        #fill shared memory
        # 0 - play
        # 1 - pause
        # 2 - stop
        self.sens_datas[0] = 0
        # set simu. speed
        self.sens_datas[1] = self.par.speed_slider.sajat_val
        # set simu. state: 0: still run, 1: run finished
        self.sens_datas[2] = 0
        # sinkhez erkezo keretek szama
        self.sens_datas[3] = 0
        # number of komm. pairs
        self.sens_datas[4] = 0
        # set the alive array, wakeful: 1, sleep: 0
        self.wakefulArray = self.sens_datas[self.simGlobParaCount:self.numProc + self.simGlobParaCount]
        # wfp100 = self.glob_paras[par.WAKEFULPROB] / 100
        # for i in range(self.numProc):
            # self.wakefulArray[i] = (1 if random.random() <= wfp100 else 0)
        # set the battery array
        # self.isEnhancedSensors = [False] * self.numProc
        self.isEnhancedSensors = self.par.isEnhSens.copy()
        self.basicen = self.glob_paras[self.par.BATTEN]
        self.battCapArray = self.sens_datas[self.numProc + self.simGlobParaCount:2 * self.numProc + self.simGlobParaCount]
        self.battCapArray[:] = self.par.bCArray
        self.initBattCapArray = self.battCapArray.copy()
        self.procData = self.sens_datas[self.simGlobParaCount + 2 * self.numProc:].reshape(self.itemPerProc, self.numProc)
        self.procData.fill(0)
        # get child shared memory
        cbuf, mm = win.Shared_query(1 if rank == 0 else 0)
        contShCount = 8
        oneContSize = 4000
        self.cpSize = 10000
        sizeChildShMem = self.cpSize + contShCount * oneContSize
        childMem = np.ndarray(buffer=cbuf, dtype='d', shape=(sizeChildShMem))
        self.commPairMem = childMem[:self.cpSize]
        self.grafShMem = [0] * contShCount
        for i in range(contShCount):
            self.grafShMem[i] = childMem[self.cpSize + i * oneContSize:self.cpSize + (i + 1) * oneContSize]
        # release the subprocesses
        self.universe.Barrier()
        
        # start draw timer
        self.timer.start(self.drawInterval)
    
    @Slot()
    def drawData(self):
        # set timer
        if self.remainderTime != 0:
            self.timer.start(self.drawInterval)
            self.remainderTime = 0
        # set sim. speed
        if self.sens_datas[1] != self.par.speed_slider.sajat_val:
            self.sens_datas[1] = self.par.speed_slider.sajat_val
        # draw data
        # for i in range(self.numProc):
        #     print("proc:", i)
        #     print("own to sink:", self.procData[i][0], "own to nb:", self.procData[i][1], "fw to sink:", self.procData[i][2], "fw to nb:", self.procData[i][3])
        #     print(flush=True)
        if self.par.qbg.checkedId() == 1 and self.par.autoGrDr.isChecked():
            self.par.viewChart()
        else:
            self.par.sharedMinMax()
        # if sim ended then stop
        if self.sens_datas[2] == 1:
            self.par.do_stop(True)
    
    def contin(self):
        self.sens_datas[0] = 0
        if self.remainderTime != 0:
            self.timer.start(self.remainderTime)
        else:
            self.timer.start(self.drawInterval)
    
    def pause(self):
        self.sens_datas[0] = 1
        self.remainderTime = self.timer.remainingTime()
        self.timer.stop()
    
    def stop(self):
        self.sens_datas[0] = 2
        self.timer.stop()
        # print("Sinkhez erkezo keretek szama:", self.sens_datas[3])
        self.sens_datas = self.sens_datas.copy()
        self.wakefulArray = self.sens_datas[self.simGlobParaCount:self.numProc + self.simGlobParaCount]
        self.wakefulArray.fill(1)
        self.battCapArray = self.sens_datas[self.numProc + self.simGlobParaCount:2 * self.numProc + self.simGlobParaCount]
        self.procData = self.sens_datas[self.simGlobParaCount + 2 * self.numProc:].reshape(self.itemPerProc, self.numProc)
        hops = self.universe.recv(source=MPI.ANY_SOURCE, tag=1)
        sources = self.universe.recv(source=MPI.ANY_SOURCE, tag=1)
        # print("hops:", hops)
        # print("sources:", sources)
        self.universe.Disconnect()
        self.comm.Disconnect()
    
    def renameSeries(self, nev):
        for i in range(7):
            self.grafSeries[i].setName(nev)
    
    def __getstate__(self):
        grafPoints = []
        for gs in self.grafSeries:
            grafPoints.append(gs.points())
        return { 'modulename': self.par.routing_alg_names[self.moduleid], 'isGroupRun': self.isGroupRun, 'runid': self.runid, 'glob_paras': self.glob_paras, 'spec_paras': self.spec_paras, 'sens_datas': self.sens_datas, 'grafSeries': grafPoints, 'procData': self.procData, 'wakefulArray': self.wakefulArray, 'battCapArray': self.battCapArray, 'isEnhancedSensors': self.isEnhancedSensors, 'initBattCapArray': self.initBattCapArray, 'numProc': self.numProc, 'itemPerProc': self.itemPerProc, 'basicen': self.basicen, 'aliveThres': self.aliveThres, 'statMax': self.statMax, 'statMin': self.statMin, 'simGlobParaCount': self.simGlobParaCount, 'locOfSens': self.locOfSens, 'sensSug': self.sensSug, 'hatotav': self.hatotav }


class Sim_widget(QWidget):
    def __init__(self, parent=None):
        super(Sim_widget, self).__init__(parent)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.volt_elso = False
        # self.awakeNPen = QPen(Qt.green)
        # self.awakeNPen.setWidth(4)
        # self.awakeEPen = QPen(Qt.cyan)
        # self.awakeEPen.setWidth(4)
        # self.sleepPen = QPen(Qt.yellow)
        # self.sleepPen.setWidth(4)
        # self.deathPen = QPen(Qt.black)
        # self.deathPen.setWidth(4)
        # self.crossPen = QPen(Qt.black)
        # self.crossPen.setWidth(4)
        self.sleepBrush = QBrush(Qt.black, bs=Qt.Dense3Pattern)
        self.nyp = QPen(Qt.black)
        self.nyp.setWidth(2)
        self.dotLine = QPen(Qt.black)
        self.dotLine.setWidth(4)
        self.dotLine.setCapStyle(Qt.FlatCap)
        self.dotLine.setStyle(Qt.DotLine)
        self.arcPen = QPen(Qt.darkGreen)
        self.arcPen.setWidth(4)
        self.d0arcPen = QPen(Qt.darkGray)
        self.d0arcPen.setStyle(Qt.DashLine)
        self.d0arcPen.setWidth(4)
        self.bordPen = QPen("#a0a0a0")
        self.bordPen.setWidth(4)
        self.sens_sug = 0.01
        self.sens_count = 0
        self.triTalpRate = 0.1
        self.cmagma = [ [0x00, 0x00, 0x03], [0x06, 0x06, 0x1d], [0x16, 0x0f, 0x3b], [0x29, 0x11, 0x5a], [0x40, 0x13, 0x73], [0x56, 0x17, 0x7d], [0x6b, 0x1d, 0x81], [0x80, 0x25, 0x82], [0x95, 0x2c, 0x80], [0xab, 0x33, 0x7c], [0xc0, 0x3a, 0x76], [0xd6, 0x45, 0x6c], [0xe8, 0x53, 0x62], [0xf4, 0x68, 0x5c], [0xfa, 0x81, 0x5f], [0xfd, 0x9a, 0x69], [0xfe, 0xb3, 0x7b], [0xfe, 0xcc, 0x8f], [0xfd, 0xe4, 0xa6], [0xfc, 0xfd, 0xbf] ]
        self.egytizenkilenced = 1 / 19
        self.arc_idx = -1
        self.feherSzin = QColor(255, 255, 255)
        self.feketeSzin = QColor(0, 0, 0)
        self.jelFont = QFont()
        self.jelFont.setPointSize(12)
        self.valLabel = QLabel("", self)
        self.valLabel.setObjectName("valLabel")
        self.valLabel.hide()
        self.valxoff = -36
        self.valyoff = -14
    
    def sizeHint(self):
        return QtCore.QSize(2000, 1000)
    
    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        par = self.parentWidget().parentWidget()
        mx = event.pos().x()
        my = event.pos().y()
        sug = int(self.sens_sug * self.height())
        sug *= sug
        kulx = mx - 9
        kuly = my - 9
        if kulx * kulx + kuly * kuly < sug:
            self.arc_idx = -2
            self.update()
            return
        for i in range(self.sens_count):
            kulx = mx - par.sensor_locs[i][0]
            kuly = my - par.sensor_locs[i][1]
            if kulx * kulx + kuly * kuly < sug:
                if event.button() == Qt.RightButton:
                    if par.is_act_play or par.natEnd or par.isSensDataShown:
                        dispidx = par.disp_combobox.currentIndex()
                        aVal = 0
                        if dispidx == 0:
                            if par.nrun.battCapArray[i] != -1:
                                aVal = (par.nrun.battCapArray[i] / par.nrun.initBattCapArray[i]) * 100
                            else:
                                aVal = 0
                        elif dispidx == 1:
                            aVal = par.nrun.procData[6][i]
                        elif dispidx == 2:
                            aVal = par.nrun.procData[9][i]
                        elif dispidx == 3:
                            aVal = par.nrun.procData[7][i]
                        elif dispidx == 4:
                            aVal = par.nrun.procData[0][i] + par.nrun.procData[1][i]
                        elif dispidx == 5:
                            aVal = par.nrun.procData[2][i] + par.nrun.procData[3][i]
                        elif dispidx == 6:
                            aVal = par.nrun.procData[0][i] + par.nrun.procData[1][i] + par.nrun.procData[2][i] + par.nrun.procData[3][i]
                        elif dispidx == 7:
                            aVal = (par.nrun.procData[8][i] / par.nrun.initBattCapArray[i]) * 100
                        elif dispidx == 8:
                            aVal = par.nrun.procData[4][i]
                        elif dispidx == 9:
                            aVal = par.nrun.procData[5][i]
                        elif dispidx == 10:
                            aVal = par.nrun.procData[10][i]
                        elif dispidx == 11:
                            aVal = (0 if par.nrun.procData[9][i] == 0 else par.nrun.procData[11][i] / par.nrun.procData[9][i])
                        elif dispidx == 12:
                            nev = par.nrun.procData[0][i] + par.nrun.procData[1][i]
                            aVal = (0 if nev == 0 else 100 * par.nrun.procData[9][i] / nev)
                        elif dispidx == 13:
                            partSum = par.nrun.procData[0][i] + par.nrun.procData[1][i] + par.nrun.procData[2][i] + par.nrun.procData[3][i]
                            nev = partSum + par.nrun.procData[4][i] + par.nrun.procData[5][i] + par.nrun.procData[10][i]
                            aVal = (0 if nev == 0 else 100 * partSum / nev)
                        self.valLabel.setText("{:.6g}".format(aVal))
                        self.valLabel.move(par.sensor_locs[i][0] + self.valxoff, par.sensor_locs[i][1] + self.valyoff)
                        # if not self.valLabel.isVisible():
                        self.valLabel.hide()
                        self.valLabel.show()
                elif self.arc_idx != i:
                    self.arc_idx = i
                    self.update()
                return
        self.arc_idx = -1
        if self.valLabel.isVisible():
            self.valLabel.hide()
        self.update()
    
    def paintEvent(self, event):
        par = self.parentWidget().parentWidget()
        wi = self.width()
        he = wi / 2
        if par.sens_draw_en:
            par.sens_draw_en = False
            self.volt_elso = True
            self.sens_count = par.glob_para_controls[par.NOSENSORS].sajat_val
            self.sens_sug = 0.01 + ((par.glob_para_controls[par.NOSENSORS].sajat_max - self.sens_count) / (par.glob_para_controls[par.NOSENSORS].sajat_max - par.glob_para_controls[par.NOSENSORS].sajat_min)) / 100
            sensor_r = int(self.sens_sug * he)
            offs = sensor_r + 11
            # genw = wi - sensor_r * 2 - 22
            # genh = he - sensor_r * 2 - 22
            genw = wi - 2 * offs
            genh = he - 2 * offs
            xt = (genw / math.sqrt(self.sens_count / (genh / genw))) * 0.786
            par.sensor_locs = []
            while len(par.sensor_locs) < self.sens_count:
                par.sensor_locs = poisson_disk.poisson_disc_samples(genw, genh, xt, 30)
            par.sensor_locs = par.sensor_locs[:self.sens_count]
            for i in range(self.sens_count):
                par.sensor_locs[i] = int(offs + par.sensor_locs[i][0]), int(offs + par.sensor_locs[i][1])
            #     sx = int(offs + random.random() * genw)
            #     sy = int(offs + random.random() * genh)
            #     par.sensor_locs.append([sx, sy])
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(self.bordPen)
        painter.setBrush(self.feherSzin)
        painter.drawRect(7, 7, wi - 14, he - 14)
        
        # draw jelmagyarazat
        hatter = QPixmap()
        hatter.load(":/icons/bee_swarm.png")
        hatter.scaled(wi, self.height() - he, aspectMode=Qt.KeepAspectRatioByExpanding)
        painter.drawPixmap(0, he, hatter)
        jelh = (self.height() - he) / 2
        jelFRH = jelh + he
        jelTextW = 0.11 * wi
        jelTextH = 2 * jelh
        jelTextY = jelFRH - 0.015 * he
        lineTextKoz = 0.01 * wi
        fLineX = (0.2 * wi) * 0.1666
        lineLen = 0.04 * wi
        distLines = lineLen + lineTextKoz + jelTextW + fLineX
        flinexPlusLinelen = fLineX + lineLen
        painter.setFont(self.jelFont)
        painter.setPen(self.feketeSzin)
        painter.drawText(flinexPlusLinelen + lineTextKoz, jelTextY, jelTextW, jelTextH, Qt.AlignLeft, "Normal node")
        painter.drawText(flinexPlusLinelen + distLines + lineTextKoz, jelTextY, jelTextW, jelTextH, Qt.AlignLeft, "Enhanced node")
        painter.drawText(flinexPlusLinelen + distLines * 2 + lineTextKoz, jelTextY, jelTextW, jelTextH, Qt.AlignLeft, "Sleeping node")
        # jelSRH = 3 * jelh + he
        # jelTextY = jelSRH - 0.013 * he
        painter.drawText(flinexPlusLinelen + distLines * 3 + lineTextKoz, jelTextY, jelTextW, jelTextH, Qt.AlignLeft, "Dead node")
        # painter.drawText(flinexPlusLinelen + distLines * 4 + lineTextKoz, jelTextY, jelTextW, jelTextH, Qt.AlignLeft, "D<sub>0</sub> distance from sink")
        painter.save()
        painter.translate(QPointF(flinexPlusLinelen + distLines * 4 + lineTextKoz, jelTextY - 6))
        td = QTextDocument()
        td.setHtml('<font size="5">D<sub>0</sub> distance from the sink</font>')
        td.drawContents(painter)
        painter.restore()
        jelRectY = jelFRH - 5
        painter.setBrush(Qt.green)
        painter.drawRect(fLineX, jelRectY, lineLen, 10)
        painter.setBrush(Qt.cyan)
        painter.drawRect(fLineX + distLines, jelRectY, lineLen, 10)
        painter.setBrush(self.sleepBrush)
        painter.drawRect(fLineX + distLines * 2, jelRectY, lineLen, 10)
        # plus sign
        painter.setPen(self.nyp)
        painter.drawLine(fLineX + distLines * 3 + lineLen * 0.125, jelFRH, fLineX + distLines * 3 + lineLen * 0.875, jelFRH)
        rty = fLineX + distLines * 3 + 0.5 * lineLen
        painter.drawLine(rty, jelFRH - lineLen * 0.375, rty, jelFRH + lineLen * 0.375)
        painter.setPen(self.d0arcPen)
        painter.drawLine(fLineX + distLines * 4, jelFRH, distLines * 4 + flinexPlusLinelen, jelFRH)
        
        # ha mar le van generalva a locs_sens
        if self.volt_elso:
            sensor_r = int(self.sens_sug * he)
            if par.under_play or par.is_act_play:
                pnsd = int(par.nrun.sens_datas[4] * 2)
                if not par.under_play:
                    par.nrun.offsetCommPair = par.oldOffsetCommPair
                    pnsd = par.oldPnsd
                # print("db", par.nrun.sens_datas[4])
                # print("csm", par.nrun.commPairMem)
                if par.nrun.offsetCommPair > pnsd:
                    par.nrun.offsetCommPair -= par.nrun.cpSize
                    # print("yeah")
                painter.setPen(self.dotLine)
                for i in range(par.nrun.offsetCommPair, pnsd, 2):
                    xr = int(par.nrun.commPairMem[i])
                    tz = int(par.nrun.commPairMem[i + 1])
                    if xr < 0 or tz < 0:
                        xr = -xr
                        tz = -tz
                    if tz != par.nrun.numProc:
                        painter.drawLine(par.sensor_locs[xr][0], par.sensor_locs[xr][1], par.sensor_locs[tz][0], par.sensor_locs[tz][1])
                    else:
                        painter.drawLine(par.sensor_locs[xr][0], par.sensor_locs[xr][1], 12, 12)
                painter.setPen(QtCore.Qt.NoPen)
                triangle = QPolygon()
                for i in range(par.nrun.offsetCommPair, pnsd, 2):
                    triangle.clear()
                    xr = int(par.nrun.commPairMem[i])
                    tz = int(par.nrun.commPairMem[i + 1])
                    # if xr == tz:
                    #     print("egyenlok", xr)
                    munar = 1
                    mutar = 1
                    if xr < 0 or tz < 0:
                        xr = -xr
                        tz = -tz
                        munar = 0.5
                        mutar = 1.2
                        painter.setBrush(Qt.darkGreen)
                    else:
                        painter.setBrush(self.feketeSzin)
                    hx = par.sensor_locs[xr][0]
                    hy = par.sensor_locs[xr][1]
                    h2x = 9; h2y = 9
                    if tz != par.nrun.numProc:
                        h2x = par.sensor_locs[tz][0]
                        h2y = par.sensor_locs[tz][1]
                    ykul = h2y - hy
                    xkul = h2x - hx
                    tav = math.sqrt(xkul * xkul + ykul * ykul)
                    egysx = xkul / tav
                    egysy = ykul / tav
                    u = (tav - sensor_r)
                    hegyx = hx + egysx * u
                    hegyy = hy + egysy * u
                    triangle.append(QPoint(hegyx, hegyy))
                    norm1x = -egysy
                    norm1y = egysx
                    norm2x = egysy
                    norm2y = -egysx
                    u = u - sensor_r * 2 * mutar
                    ntav = sensor_r * munar
                    talpx = hx + egysx * u
                    talpy = hy + egysy * u
                    norm1x = talpx + norm1x * ntav
                    norm1y = talpy + norm1y * ntav
                    norm2x = talpx + norm2x * ntav
                    norm2y = talpy + norm2y * ntav
                    triangle.append(QPoint(norm1x, norm1y))
                    triangle.append(QPoint(norm2x, norm2y))
                    painter.drawPolygon(triangle)
                par.oldOffsetCommPair = par.nrun.offsetCommPair
                par.oldPnsd = pnsd
                par.nrun.offsetCommPair = pnsd
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QColor(0, 0, 255))
            sink_d = int(0.05 * he)
            sink_r = int(0.025 * he)
            offs = 9
            painter.drawPie( -sink_r + offs, -sink_r + offs, sink_d, sink_d, 270 * 16, 90 * 16)
            # draw d0 arc
            painter.setPen(self.d0arcPen)
            factd0 = (he - 2 * offs) / par.glob_para_controls[par.AREARECTH].sajat_val
            painter.drawArc(offs - factd0, offs - factd0, factd0 * 2, factd0 * 2, 270 * 16, 90 * 16)
            #--------------
            if par.is_act_play or par.natEnd or par.isSensDataShown:
                dispidx = par.disp_combobox.currentIndex()
                cmin = par.nrun.statMin
                cmax = par.nrun.statMax
                if cmin == cmax:
                    cmax = cmin + 1
                pointerToValues = 0
                if dispidx == 0:
                    pointerToValues = [0] * self.sens_count
                    for xd in range(self.sens_count):
                        if par.nrun.battCapArray[xd] != -1:
                            pointerToValues[xd] = (par.nrun.battCapArray[xd] / par.nrun.initBattCapArray[xd]) * 100
                        else:
                            pointerToValues[xd] = 0
                elif dispidx == 1:
                    pointerToValues = par.nrun.procData[6]
                elif dispidx == 2:
                    pointerToValues = par.nrun.procData[9]
                elif dispidx == 3:
                    pointerToValues = par.nrun.procData[7]
                elif dispidx == 4:
                    pointerToValues = [0] * self.sens_count
                    for xd in range(self.sens_count):
                        pointerToValues[xd] = par.nrun.procData[0][xd] + par.nrun.procData[1][xd]
                elif dispidx == 5:
                    pointerToValues = [0] * self.sens_count
                    for xd in range(self.sens_count):
                        pointerToValues[xd] = par.nrun.procData[2][xd] + par.nrun.procData[3][xd]
                elif dispidx == 6:
                    pointerToValues = [0] * self.sens_count
                    for xd in range(self.sens_count):
                        pointerToValues[xd] = par.nrun.procData[0][xd] + par.nrun.procData[1][xd] + par.nrun.procData[2][xd] + par.nrun.procData[3][xd]
                elif dispidx == 7:
                    pointerToValues = [0] * self.sens_count
                    for xd in range(self.sens_count):
                        pointerToValues[xd] = (par.nrun.procData[8][xd] / par.nrun.initBattCapArray[xd]) * 100
                elif dispidx == 8:
                    pointerToValues = par.nrun.procData[4]
                elif dispidx == 9:
                    pointerToValues = par.nrun.procData[5]
                elif dispidx == 10:
                    pointerToValues = par.nrun.procData[10]
                elif dispidx == 11:
                    pointerToValues = [0] * self.sens_count
                    for xd in range(self.sens_count):
                        pointerToValues[xd] = (0 if par.nrun.procData[9][xd] == 0 else par.nrun.procData[11][xd] / par.nrun.procData[9][xd])
                elif dispidx == 12:
                    pointerToValues = [0] * self.sens_count
                    for xd in range(self.sens_count):
                        nev = par.nrun.procData[0][xd] + par.nrun.procData[1][xd]
                        pointerToValues[xd] = (0 if nev == 0 else 100 * par.nrun.procData[9][xd] / nev)
                elif dispidx == 13:
                    pointerToValues = [0] * self.sens_count
                    for xd in range(self.sens_count):
                        partSum = par.nrun.procData[0][xd] + par.nrun.procData[1][xd] + par.nrun.procData[2][xd] + par.nrun.procData[3][xd]
                        nev = partSum + par.nrun.procData[4][xd] + par.nrun.procData[5][xd] + par.nrun.procData[10][xd]
                        pointerToValues[xd] = (0 if nev == 0 else 100 * partSum / nev)
                    
                
                painter.setPen(self.nyp)
                for i in range(self.sens_count):
                    # ha alszik es el
                    if not par.nrun.wakefulArray[i] and par.nrun.battCapArray[i] != -1:
                        painter.setBrush(self.sleepBrush)
                        hatkor_r = sensor_r + 12
                        painter.drawEllipse(QPoint(par.sensor_locs[i][0], par.sensor_locs[i][1]), hatkor_r, hatkor_r)
                    # ha enhanced, ha nem
                    if par.nrun.isEnhancedSensors[i]:
                        painter.setBrush(Qt.cyan)
                    else:
                        painter.setBrush(Qt.green)
                    hatkor_r = sensor_r + 6
                    painter.drawEllipse(QPoint(par.sensor_locs[i][0], par.sensor_locs[i][1]), hatkor_r, hatkor_r)
                if par.skala_widget.isMagma:
                    for i in range(self.sens_count):
                        car = 1 - (pointerToValues[i] - cmin) / (cmax - cmin)
                        if car < 0:
                            car = 0
                        elif car > 1:
                            car = 1
                        zx = int(car / self.egytizenkilenced)
                        if zx == 19:
                            zx = 18
                        arany = (car - zx * self.egytizenkilenced) / self.egytizenkilenced
                        sqc = QColor(int((1 - arany) * self.cmagma[zx][0] + arany * self.cmagma[zx + 1][0]), int((1 - arany) * self.cmagma[zx][1] + arany * self.cmagma[zx + 1][1]), int((1 - arany) * self.cmagma[zx][2] + arany * self.cmagma[zx + 1][2]))
                        painter.setBrush(sqc)
                        painter.drawEllipse(QPoint(par.sensor_locs[i][0], par.sensor_locs[i][1]), sensor_r, sensor_r)
                else:
                    for i in range(self.sens_count):
                        car = 255 - int(((pointerToValues[i] - cmin) / (cmax - cmin)) * 255)
                        sqc = QColor(car, car, car)
                        painter.setBrush(sqc)
                        painter.drawEllipse(QPoint(par.sensor_locs[i][0], par.sensor_locs[i][1]), sensor_r, sensor_r)
                # kereszt rajzolas ha nem el
                zt = sensor_r * 1.5
                for i in range(self.sens_count):
                    if par.nrun.battCapArray[i] == -1:
                        # vertical line
                        painter.drawLine(par.sensor_locs[i][0], par.sensor_locs[i][1] - zt, par.sensor_locs[i][0], par.sensor_locs[i][1] + zt)
                        # horizontal line
                        painter.drawLine(par.sensor_locs[i][0] - zt, par.sensor_locs[i][1], par.sensor_locs[i][0] + zt, par.sensor_locs[i][1])
            else:
                painter.setPen(Qt.NoPen)
                painter.setBrush(self.feketeSzin)
                for i in range(self.sens_count):
                    painter.drawEllipse(QPoint(par.sensor_locs[i][0], par.sensor_locs[i][1]), sensor_r, sensor_r)
                par.sensd0 = (he - 2 * offs) / par.glob_para_controls[par.AREARECTH].sajat_val
                hatotav = 25**(1 / par.glob_para_controls[par.BPATHLOSS].sajat_val) * par.sensd0
                if hatotav != par.hatotav:
                    par.hatotav = hatotav
            if self.arc_idx != -1:
                painter.setBrush(Qt.green)
                hatotav = par.hatotav
                ohatotav = hatotav
                hatotav *= hatotav
                ox = 9
                oy = 9
                if self.arc_idx != -2 and self.arc_idx < self.sens_count:
                    ox = par.sensor_locs[self.arc_idx][0]
                    oy = par.sensor_locs[self.arc_idx][1]
                for i in range(self.sens_count):
                    kulx = ox - par.sensor_locs[i][0]
                    kuly = oy - par.sensor_locs[i][1]
                    if kulx * kulx + kuly * kuly <= hatotav:
                        painter.drawEllipse(QPoint(par.sensor_locs[i][0], par.sensor_locs[i][1]), sensor_r, sensor_r)
                if self.arc_idx != -2:
                    painter.setBrush(Qt.darkGreen)
                    painter.drawEllipse(QPoint(ox, oy), sensor_r, sensor_r)
                painter.setPen(self.arcPen)
                painter.drawArc(ox - ohatotav, oy - ohatotav, ohatotav * 2, ohatotav * 2, 0, 360 * 16)
            

class Skala_widget(QWidget):
    def __init__(self, parent=None):
        super(Skala_widget, self).__init__(parent)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.munits = ["%", "rounds", "pcs.", "pcs.", "pcs.", "pcs.", "pcs.", "%", "pcs.", "pcs.", "pcs.", "hop", "%", "%"]
        self.munitPen = QPen(Qt.black)
        self.bePen = QPen(QColor(0, 0, 0, 255))
        self.bePen.setWidth(3)
        self.munitFont = QFont()
        self.munitFont.setPointSize(16)
        self.beosztFont = QFont()
        self.beosztFont.setPointSize(12)
        self.isMagma = True
        self.gradi = QLinearGradient(0, 0, 50, 50)
        self.gradi.setColorAt(0, "#000003")
        self.gradi.setColorAt(1 / 19, "#06061d")
        self.gradi.setColorAt(2 / 19, "#160f3b")
        self.gradi.setColorAt(3 / 19, "#29115a")
        self.gradi.setColorAt(4 / 19, "#401373")
        self.gradi.setColorAt(5 / 19, "#56177d")
        self.gradi.setColorAt(6 / 19, "#6b1d81")
        self.gradi.setColorAt(7 / 19, "#802582")
        self.gradi.setColorAt(8 / 19, "#952c80")
        self.gradi.setColorAt(9 / 19, "#ab337c")
        self.gradi.setColorAt(10 / 19, "#c03a76")
        self.gradi.setColorAt(11 / 19, "#d6456c")
        self.gradi.setColorAt(12 / 19, "#e85362")
        self.gradi.setColorAt(13 / 19, "#f4685c")
        self.gradi.setColorAt(14 / 19, "#fa815f")
        self.gradi.setColorAt(15 / 19, "#fd9a69")
        self.gradi.setColorAt(16 / 19, "#feb37b")
        self.gradi.setColorAt(17 / 19, "#fecc8f")
        self.gradi.setColorAt(18 / 19, "#fde4a6")
        self.gradi.setColorAt(1, "#fcfdbf")
        self.gradbw = QLinearGradient(0, 0, 50, 50)
        self.gradbw.setColorAt(0, Qt.black)
        self.gradbw.setColorAt(1, Qt.white)
        
    def sizeHint(self):
        return QtCore.QSize(2000, 1000)
    
    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if self.isMagma:
            self.isMagma = False
        else:
            self.isMagma = True
        self.update()
        self.parentWidget().parentWidget().sim_widget.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        he = self.height()
        wi = self.width()
        gradrectX = int(0.4 * wi)
        gradrectY = int(0.06 * he)
        gradrectWi = int(0.46 * wi)
        gradrectHe = int(0.9 * he)
        par = self.parentWidget().parentWidget()
        kivi = par.disp_combobox.currentIndex()
        if kivi != -1:
            painter.setPen(self.munitPen)
            painter.setFont(self.munitFont)
            painter.drawText(gradrectX, int(0.01 * he), gradrectWi, int(0.04 * he), Qt.AlignCenter, self.munits[kivi])
        painter.setPen(QtCore.Qt.NoPen)
        if self.isMagma:
            self.gradi.setStart(gradrectX, gradrectY)
            self.gradi.setFinalStop(gradrectX + gradrectWi, gradrectY + gradrectHe)
            painter.setBrush(self.gradi)
        else:
            self.gradbw.setStart(gradrectX, gradrectY)
            self.gradbw.setFinalStop(gradrectX + gradrectWi, gradrectY + gradrectHe)
            painter.setBrush(self.gradbw)
        painter.drawRect(gradrectX, gradrectY, gradrectWi, gradrectHe)
        painter.setPen(self.munitPen)
        painter.setFont(self.beosztFont)
        beoszt_sx = int(gradrectX - 0.05 * wi)
        text_wi = int(beoszt_sx - 0.02 * wi)
        text_he = int(0.02 * he)
        text_mi = int(0.013 * he)
        beCount = 20
        beoszt_unit = gradrectHe / beCount
        if par.is_act_play or par.natEnd or par.isSensDataShown:
            pformat = ('%.0f' if kivi != 11 and kivi != 12 else '%.2f')
            cmin = par.nrun.statMin
            cmax = par.nrun.statMax
            if kivi != 11 and kivi != 12:
                zd = cmax - cmin
                if zd < 0.0001:
                    zd = 1
                if zd < 20 and zd >= 1:
                    beCount = zd
                    beoszt_unit = gradrectHe / beCount
            cunit = (cmax - cmin) / beCount
            for i in range(int(beCount) + 1):
                uy = int(gradrectY + i * beoszt_unit)
                painter.drawText(0, uy - text_mi, text_wi, text_he, Qt.AlignRight, pformat % (cmin + (beCount - i) * cunit))
        beoszt_vx = int(gradrectX + 0.02 * wi)
        painter.setPen(self.bePen)
        for i in range(int(beCount) + 1):
            uy = int(gradrectY + i * beoszt_unit)
            painter.drawLine(beoszt_sx, uy, beoszt_vx, uy)
        
class CheckBox(QCheckBox):
    baseColor = QPalette(QColor("#f0f0f0"))
    selColor = QPalette(QColor("#AFFFF2"))
    def __init__(self, name, moduleid, runid, par, szimrun, parent=None):
        super(CheckBox, self).__init__(name)
        self.moduleid = moduleid
        self.runid = runid
        self.par = par
        self.szimrun = szimrun
        # self.setPalette()
        self.setAutoFillBackground(True)
        self.setFocusPolicy(Qt.ClickFocus)
    
    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.RightButton:
            if self.par.prevSelRun != 0 and self.par.prevSelRun != self:
                self.par.prevSelRun.setPalette(CheckBox.baseColor)
            self.par.prevSelRun = self
            self.setPalette(CheckBox.selColor)
            self.par.listParasOfRun(self.moduleid, self.szimrun)
            
    def keyPressEvent(self, event):
        key = event.key()
        if key != Qt.Key_Tab:
            if key == Qt.Key_Backspace:
                self.setText(self.text()[:-1])
                self.szimrun.renameSeries(self.text())
            elif key == Qt.Key_Delete:
                self.setText("")
            elif key == Qt.Key_Up:
                tt = self.par.laysPerModul[self.moduleid].layout()
                saj_id = tt.indexOf(self)
                if saj_id > 1:
                    tt.removeWidget(self)
                    tt.insertWidget(saj_id - 1, self)
            elif key == Qt.Key_Down:
                tt = self.par.laysPerModul[self.moduleid].layout()
                saj_id = tt.indexOf(self)
                if saj_id < self.par.runCountPerModul[self.moduleid]:
                    tt.removeWidget(self)
                    tt.insertWidget(saj_id + 1, self)
            else:
                self.setText(self.text() + event.text())
                self.szimrun.renameSeries(self.text())
        else:
            super().keyPressEvent(event)
            

class ComboBox(QComboBox):
    def __init__(self, parlabel, alaptext, defindex, parent=None):
        super(ComboBox, self).__init__(parent)
        self.parlabel = parlabel
        self.alaptext = alaptext
        self.defindex = defindex
        self.currentIndexChanged[str].connect(self.editLabel)
        
    @Slot()
    def editLabel(self, atxt):
        self.parlabel.setText(self.alaptext + ": <b>" + atxt + "</b>")

class Slider(QSlider):
    beosztas = 10000
    def __init__(self, ptype, mini, maxi, val, parlabel, alaptext, parent=None):
        super(Slider, self).__init__(parent)
        self.parlabel = parlabel
        self.alaptext = alaptext
        self.sajat_min = mini
        self.sajat_max = maxi
        self.sajat_val = val
        self.ori_val = val
        self.tartomany = maxi - mini
        self.sajat_step = self.tartomany / Slider.beosztas
        self.formstr = ""
        self.ptype = ptype
        if self.ptype == 0:
            self.formstr = "{:d}"
        else:
            if self.tartomany < 5:
                self.formstr = "{:.2f}"
            else:
                self.formstr = "{:.1f}"
        self.ispaint = False
        self.firstRun = True
        self.groove_start_label = QLabel(str(mini))
        self.groove_start_label.setParent(self)
        self.groove_end_label = QLabel(str(maxi))
        self.groove_end_label.setParent(self)
        self.handle_label = QLabel()
        self.handle_label.setParent(self)
        self.setSingleStep(1)
        self.setOrientation(Qt.Horizontal)
        self.setMinimum(0)
        self.setMaximum(Slider.beosztas)
        zz = (self.sajat_val - self.sajat_min) / self.sajat_step
        kk = int(zz) + 0.5
        if zz < kk:
            zz = int(zz)
        else:
            zz = int(zz) + 1
        self.orislvalue = zz
        self.setValue(zz)
    
    def sliderChange(self, change):
        super().sliderChange(change)
        if change == QAbstractSlider.SliderValueChange:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            pp = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle)
            sr = pp.topLeft()
            if not self.ispaint:
                zz = self.value() * self.sajat_step + self.sajat_min
                if self.ptype == 0:
                    kk = int(zz) + 0.5
                    if zz < kk:
                        self.sajat_val = int(zz)
                    else:
                        self.sajat_val = int(zz) + 1
                else:
                    self.sajat_val = zz
            else:
                self.sajat_val = self.ori_val
            if self.sajat_val > self.sajat_max:
                self.sajat_val = self.sajat_max
            sval = self.formstr.format(self.sajat_val)
            self.handle_label.setText(sval)
            self.handle_label.adjustSize()
            wq = (self.handle_label.width() - pp.width()) * 0.4
            if wq < 0:
                wq = 0
            self.handle_label.move(sr.x() + 2 - (self.value() / self.maximum()) * (self.handle_label.width() / 2 + wq), sr.y())
            self.parlabel.setText(self.alaptext + ": <b>" + sval + "</b>")
            if self.ispaint:
                self.ispaint = False
                sr = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove)
                self.groove_start_label.adjustSize()
                a = sr.bottomLeft()
                self.groove_start_label.move(a.x(), a.y() - self.groove_start_label.height())
                self.groove_end_label.adjustSize()
                a = sr.bottomRight()
                self.groove_end_label.move(a.x() - self.groove_end_label.width(), a.y() - self.groove_end_label.height())
            
    def paintEvent(self, event):
        super().paintEvent(event)
        if self.firstRun:
            self.firstRun = False
            self.ispaint = True
            self.sliderChange(QAbstractSlider.SliderValueChange)
            

class Widget(QWidget):
    def __init__(self, algNameListLen, parent=None):
        super(Widget, self).__init__(parent)

        self.setWindowTitle("Wireless Network simulator with Swarm Intelligence (WNSI)")
        
        # main_r_layout-------------------------------------------------------
        # swarm intelligence algorithm selector
        self.routing_alg_names = []
        self.si_select_list = QListWidget()
        self.routing_algs = []
        self.spec_para_stacked_layout = QStackedLayout()
        self.algs_spec_controls = []
        self.algs_spec_isSliders = []
        self.alg_count = 0
        self.spec_para_stacked_layout.addWidget(QWidget())
        self.spec_para_stacked_layout.setCurrentIndex(self.alg_count)
        self.si_select_list.currentRowChanged.connect(self.spec_para_stacked_layout.setCurrentIndex)
        # optimization
        opt_button = QPushButton("Optimization")
        refr_button = QPushButton("R")
        refr_button.clicked.connect(self.refresh_modules)
        def_button = QPushButton("Def")
        def_button.clicked.connect(self.set_to_default)
        opt_layout = QHBoxLayout()
        opt_layout.addWidget(opt_button, 4)
        opt_layout.addWidget(def_button, 2)
        opt_layout.addWidget(refr_button, 1)
        # si alg. spec. params
        self.spec_par_widget = QWidget()
        self.spec_par_widget.setObjectName("spw")
        self.spec_par_widget.setLayout(self.spec_para_stacked_layout)
        
        main_r_layout = QVBoxLayout()
        main_r_layout.addWidget(self.si_select_list, 3)
        main_r_layout.addLayout(opt_layout, 2)
        main_r_layout.addWidget(self.spec_par_widget, 16)
        tkp = QPixmap()
        tkp.load(":/icons/wnsi_logo.png")
        klj = QLabel()
        klj.setPixmap(tkp)
        main_r_layout.addWidget(klj, 3)
        main_r_layout.setContentsMargins(0, 0, 0, 0)
        
        # main_l_layout-------------------------------------------------------
        sim_button = QPushButton("Simulator")
        sim_button.setObjectName("simb")
        sim_button.setCheckable(True)
        sim_button.setChecked(True)
        sim_button.clicked.connect(self.switchToSimView)
        ana_button = QPushButton("Analysis")
        ana_button.setObjectName("simb")
        ana_button.setCheckable(True)
        self.qbg = QButtonGroup()
        self.qbg.addButton(sim_button)
        self.qbg.addButton(ana_button)
        self.qbg.setId(sim_button, 0)
        self.qbg.setId(ana_button, 1)
        alctrl_1_layout = QHBoxLayout()
        alctrl_1_layout.addWidget(sim_button)
        alctrl_1_layout.addWidget(ana_button)
        alctrl_1_layout.setContentsMargins(0, 0, 0, 0)
        alctrl_1_layout.setSpacing(0)
        
        slow_button = QPushButton("<")
        slow_button.setObjectName("sped")
        slow_button.clicked.connect(self.dec_speed)
        speed_label = QLabel()
        self.speed_slider = Slider(0, 1, 100, 50, speed_label, "Speed")
        zc = Slider.beosztas / self.speed_slider.tartomany
        if zc > int(zc):
            zc = int(zc) + 1
        self.speed_nov = zc
        acc_button = QPushButton(">")
        acc_button.setObjectName("sped")
        acc_button.clicked.connect(self.inc_speed)
        alctrl_2_layout = QHBoxLayout()
        alctrl_2_layout.addWidget(slow_button)
        alctrl_2_layout.addWidget(self.speed_slider)
        alctrl_2_layout.addWidget(acc_button)
        
        self.playtime_label = QLabel("00:00")
        self.playtime_label.setObjectName("playl")
        self.playtime_count = 0
        self.remainder = 0
        self.under_play = False
        self.play_timer = QTimer(self)
        self.play_timer.setTimerType(Qt.PreciseTimer)
        self.play_timer.timeout.connect(self.time_update)
        stop_button = QPushButton()
        stop_button.clicked.connect(self.do_stop)
        stop_button.setIcon(QApplication.style().standardIcon(QStyle.SP_MediaStop))
        self.play_button = QPushButton()
        self.play_button.clicked.connect(self.do_play)
        self.play_button.setIcon(QApplication.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.setEnabled(False)
        # play_button.setSizePolicy(play_button.sizePolicy().horizontalPolicy(), QSizePolicy.Minimum)
        alctrl_3_layout = QHBoxLayout()
        alctrl_3_layout.addWidget(self.playtime_label, 1)
        alctrl_3_layout.addWidget(stop_button, 1)
        alctrl_3_layout.addWidget(self.play_button, 1)
        alctrl_3_layout.setContentsMargins(0, 0, 0, 0)
        # alctrl_3_layout.setSizeConstraint(QLayout.SetFixedSize)
        # speed_label.setSizePolicy(play_button.sizePolicy().horizontalPolicy(), QSizePolicy.Fixed)
        self.locsens_button = QPushButton("Locate sensors")
        self.locsens_button.clicked.connect(self.locate_sensors)
        
        control_layout = QVBoxLayout()
        control_layout.addStretch()
        control_layout.addLayout(alctrl_1_layout)
        control_layout.addWidget(speed_label)
        control_layout.addLayout(alctrl_2_layout)
        control_layout.addWidget(self.locsens_button)
        control_layout.addLayout(alctrl_3_layout)
        control_layout.setContentsMargins(5, 0, 8, 0)
        w_control_layout = QWidget()
        w_control_layout.setLayout(control_layout)
        w_control_layout.setObjectName("ctrlly")
        
        self.top_para_layout = QStackedLayout()
        glob_para_layout = QGridLayout()
        glob_para_layout.setContentsMargins(0, 0, 0, 0)
        glob_para_layout.setSpacing(0)
        self.glob_para_controls = []
        
        k = QLabel()
        z = Slider(0, 1, 5, 3, k, "Area rect height (D<sub>0</sub>)")
        z.valueChanged.connect(self.commrChanged)
        self.glob_para_controls.append(z)
        al = QVBoxLayout()
        al.addWidget(k)
        al.addWidget(z)
        aw = QWidget()
        aw.setLayout(al)
        aw.setObjectName("fgridwidget")
        glob_para_layout.addWidget(aw, 0, 0, 2, 1)
        self.AREARECTH = 0
        
        k = QLabel()
        z = Slider(1, 0.1, 3, 2.5, k, "Battery energy (E<sub>0</sub>)")
        self.glob_para_controls.append(z)
        al = QVBoxLayout()
        al.addWidget(k)
        al.addWidget(z)
        aw = QWidget()
        aw.setLayout(al)
        aw.setObjectName("agridwidget")
        glob_para_layout.addWidget(aw, 2, 2, 2, 1)
        self.BATTEN = 1
        
        k = QLabel()
        z = Slider(0, 0, 100, 10, k, "Alive battery threshold (%)")
        self.glob_para_controls.append(z)
        al = QVBoxLayout()
        al.addWidget(k)
        al.addWidget(z)
        aw = QWidget()
        aw.setLayout(al)
        aw.setObjectName("fgridwidget")
        glob_para_layout.addWidget(aw, 0, 1, 2, 1)
        self.ALIBATTHRES = 2
        
        k = QLabel()
        z = Slider(0, 20, 1000, 20, k, "Number of sensors")
        self.glob_para_controls.append(z)
        al = QVBoxLayout()
        al.addWidget(k)
        al.addWidget(z)
        aw = QWidget()
        aw.setLayout(al)
        aw.setObjectName("agridwidget")
        glob_para_layout.addWidget(aw, 2, 1, 2, 1)
        self.NOSENSORS = 3
        
        k = QLabel()
        z = Slider(1, 0, 1, 0.5, k, "Alpha (metric param.)")
        self.glob_para_controls.append(z)
        al = QVBoxLayout()
        al.addWidget(k)
        al.addWidget(z)
        aw = QWidget()
        aw.setLayout(al)
        aw.setObjectName("fgridwidget")
        glob_para_layout.addWidget(aw, 0, 2, 2, 1)
        self.ALPHA = 4
        
        k = QLabel()
        z = Slider(1, 2, 4, 3, k, "b: path loss exp.")
        z.valueChanged.connect(self.commrChanged)
        self.glob_para_controls.append(z)
        al = QVBoxLayout()
        al.addWidget(k)
        al.addWidget(z)
        aw = QWidget()
        aw.setLayout(al)
        aw.setObjectName("agridwidget")
        glob_para_layout.addWidget(aw, 2, 0, 2, 1)
        self.BPATHLOSS = 5
        
        k = QLabel()
        z = Slider(0, 1, 100, 50, k, "Enhanced nodes (EN) rate (%)")
        self.glob_para_controls.append(z)
        al = QVBoxLayout()
        al.addWidget(k)
        al.addWidget(z)
        aw = QWidget()
        aw.setLayout(al)
        aw.setObjectName("fugridwidget")
        glob_para_layout.addWidget(aw, 0, 4, 2, 1)
        self.ENHNODERATE = 6
        
        k = QLabel()
        z = Slider(0, 20, 125, 75, k, "Frame length (B)")
        self.glob_para_controls.append(z)
        al = QVBoxLayout()
        al.addWidget(k)
        al.addWidget(z)
        aw = QWidget()
        aw.setLayout(al)
        aw.setObjectName("agridwidget")
        glob_para_layout.addWidget(aw, 2, 3, 2, 1)
        self.FRAMELEN = 7
        
        k = QLabel()
        z = Slider(1, 0, 1, 0.25, k, "Wakeful probability")
        self.glob_para_controls.append(z)
        al = QVBoxLayout()
        al.addWidget(k)
        al.addWidget(z)
        aw = QWidget()
        aw.setLayout(al)
        aw.setObjectName("fgridwidget")
        glob_para_layout.addWidget(aw, 0, 3, 2, 1)
        self.WAKEFULPROB = 8
        
        k = QLabel()
        z = Slider(0, 0, 200, 25, k, "Extra energy of ENs (%)")
        self.glob_para_controls.append(z)
        al = QVBoxLayout()
        al.addWidget(k)
        al.addWidget(z)
        aw = QWidget()
        aw.setLayout(al)
        glob_para_layout.addWidget(aw, 2, 4, 2, 1)
        self.EXTRAENERGY = 9
        
        k = QLabel()
        z = Slider(1, 0.01, 1, 0.5, k, "Data origin probability")
        self.glob_para_controls.append(z)
        al = QVBoxLayout()
        al.addWidget(k)
        al.addWidget(z)
        aw = QWidget()
        aw.setLayout(al)
        aw.setObjectName("fegridwidget")
        glob_para_layout.addWidget(aw, 4, 0, 2, 1)
        self.DATAORIGINPROB = 10
        
        k = QLabel()
        z = Slider(0, 1, 20, 4, k, "Iterations for meas.")
        self.glob_para_controls.append(z)
        al = QVBoxLayout()
        al.addWidget(k)
        al.addWidget(z)
        aw = QWidget()
        aw.setLayout(al)
        aw.setObjectName("fegridwidget")
        glob_para_layout.addWidget(aw, 6, 0, 2, 1)
        self.ITERFORMEAS = 11
        
        self.glob_para_len = 12
        #----------------------------------------------------------------------------------------------
        g = QVBoxLayout()
        self.isGroupRunsCh = QCheckBox("Batch run with one parameter")
        self.isGroupRunsCh.setObjectName("isgrouprch")
        self.isGroupRunsCh.stateChanged.connect(self.groupRuns)
        g.addWidget(self.isGroupRunsCh)
        
        self.grouprun_bg = QButtonGroup()
        a = QHBoxLayout()
        a.setContentsMargins(0, 5, 0, 0)
        a.setSpacing(0)
        b = QPushButton("Glob.")
        b.setObjectName("glopospec")
        b.setCheckable(True)
        b.setChecked(True)
        self.grouprun_bg.addButton(b)
        self.grouprun_bg.setId(b, 0)
        a.addWidget(b)
        b = QPushButton("Spec.")
        b.setObjectName("glopospec")
        b.setCheckable(True)
        self.grouprun_bg.addButton(b)
        self.grouprun_bg.setId(b, 1)
        a.addWidget(b)
        self.specOrGlobBW = QWidget()
        self.specOrGlobBW.setLayout(a)
        g.addWidget(self.specOrGlobBW)
        self.specOrGlobBW.hide()
        grstack = QStackedLayout()
        self.grouprun_bg.idClicked.connect(grstack.setCurrentIndex)
        self.groupRunGlobCb = QComboBox()
        self.groupRunGlobCb.addItems(["Alive battery threshold (%)", "Alpha (metric param.)", "Area rect height (Do)", "Battery energy (Eo)", "b: path loss exp.", "Data origin probability", "Enhanced nodes (EN) rate (%)", "Extra energy of ENs (%)", "Frame length (B)", "Iterations for meas. (% of popu.)", "Number of sensors", "Wakeful probability"])
        self.groupRunGlobCb.currentIndexChanged.connect(self.showGlobFromToSliders)
        grstack.addWidget(self.groupRunGlobCb)
        self.groupRunSpecCb = QComboBox()
        self.groupRunSpecCb.currentIndexChanged.connect(self.showSpecFromToSliders)
        grstack.addWidget(self.groupRunSpecCb)
        self.specOrGlobStW = QWidget()
        self.specOrGlobStW.setLayout(grstack)
        g.addWidget(self.specOrGlobStW)
        g.addStretch()
        self.specOrGlobStW.hide()
        glob_para_layout.addLayout(g, 4, 1, 1, 1)
        self.si_select_list.currentRowChanged.connect(self.uploadSpecCb)
        
        self.fromSlLabel = QLabel()
        self.toSlLabel = QLabel()
        self.stepSlLabel = QLabel()
        self.fromSlider = Slider(0, 1, 100, 50, self.fromSlLabel, "")
        self.toSlider = Slider(0, 1, 100, 50, self.toSlLabel, "")
        self.stepSlider = Slider(0, 1, 100, 50, self.stepSlLabel, "")
        self.ftoSl_layout = QVBoxLayout()
        self.ftoSl_layout.addWidget(self.fromSlLabel)
        self.ftoSl_layout.addWidget(self.fromSlider)
        self.ftoSl_layout.addWidget(self.toSlLabel)
        self.ftoSl_layout.addWidget(self.toSlider)
        self.ftoSl_layout.addStretch()
        self.ftoSl_layout_W = QWidget()
        self.ftoSl_layout_W.setLayout(self.ftoSl_layout)
        glob_para_layout.addWidget(self.ftoSl_layout_W, 4, 2, 2, 1)
        self.ftoSl_layout_W.hide()
        self.fto_step_layout = QVBoxLayout()
        self.fto_step_layout.addWidget(self.stepSlLabel)
        self.fto_step_layout.addWidget(self.stepSlider)
        self.fto_step_layout.addStretch()
        self.fto_step_layout_W = QWidget()
        self.fto_step_layout_W.setLayout(self.fto_step_layout)
        glob_para_layout.addWidget(self.fto_step_layout_W, 4, 3, 1, 1)
        self.fto_step_layout_W.hide()
        #--------------------------------------------------------------------------------------------
        wgpl = QWidget()
        wgpl.setLayout(glob_para_layout)
        scwgpl = QScrollArea()
        scwgpl.setObjectName("topscroll")
        scwgpl.setWidgetResizable(True)
        scwgpl.setWidget(wgpl)
        self.top_para_layout.addWidget(scwgpl)
        
        ana_para_layout = QGridLayout()
        ana_para_layout.setContentsMargins(0, 0, 0, 0)
        #first column-------------------------------------------------------------------------
        frow_layout = QVBoxLayout()
        # frow_layout.addWidget(QLabel("algorithms"))
        self.falrow_layout = QVBoxLayout()
        self.falrow_layout.setContentsMargins(6, 0, 0, 0)
        self.ana_alg_ch_bg = QButtonGroup()
        self.ana_alg_ch_bg.setExclusive(False)
        for i in range(self.alg_count):
            z = QCheckBox(self.routing_alg_names[i])
            self.ana_alg_ch_bg.addButton(z)
            self.ana_alg_ch_bg.setId(z, i)
            self.falrow_layout.addWidget(z)
        self.ana_alg_ch_bg.idToggled.connect(self.listRunsOfModul)
        s = QWidget()
        s.setLayout(self.falrow_layout)
        d = QScrollArea()
        d.setWidgetResizable(True)
        d.setWidget(s)
        frow_layout.addWidget(d)
        loadGrid = QGridLayout()
        ww = QPushButton("Load")
        ww.clicked.connect(self.loadRun)
        loadGrid.addWidget(ww, 0, 0, 1, 3)
        ww = QPushButton("Save")
        ww.clicked.connect(self.saveRun)
        loadGrid.addWidget(ww, 0, 3, 1, 3)
        ww = QPushButton("Del. choosed")
        ww.clicked.connect(self.delChoosed)
        loadGrid.addWidget(ww, 1, 0, 1, 4)
        ww = QPushButton("View")
        ww.clicked.connect(self.viewChart)
        loadGrid.addWidget(ww, 1, 4, 1, 2)
        frow_layout.addLayout(loadGrid)
        ana_para_layout.addLayout(frow_layout, 0, 0, 1, 1)
        #second column-------------------------------------------------------------------------
        srow_layout = QVBoxLayout()
        # z = QCheckBox("Runs")
        # z.stateChanged.connect(self.selectAll)
        p = QHBoxLayout()
        z = QPushButton("Select all runs")
        z.setObjectName("selallb")
        z.clicked.connect(self.selectAll)
        p.addWidget(z)
        z = QPushButton("Deselect all runs")
        z.setObjectName("selallb")
        z.clicked.connect(self.deSelectAll)
        p.addWidget(z)
        srow_layout.addLayout(p)
        self.salrow_layout = QVBoxLayout()
        self.runCountPerModul = []
        self.runIdCountPerModul = []
        self.laysPerModul = []
        s = QWidget()
        s.setLayout(self.salrow_layout)
        d = QScrollArea()
        d.setWidgetResizable(True)
        d.setWidget(s)
        srow_layout.addWidget(d)
        ana_para_layout.addLayout(srow_layout, 0, 1, 1, 1)
        #third column-------------------------------------------------------------------------
        throw_layout = QVBoxLayout()
        # throw_layout.addWidget(QLabel("Details"))
        self.globspec_bg = QButtonGroup()
        a = QHBoxLayout()
        a.setSpacing(0)
        b = QPushButton("Glob.")
        b.setObjectName("glopospec")
        b.setCheckable(True)
        b.setChecked(True)
        self.globspec_bg.addButton(b)
        self.globspec_bg.setId(b, 0)
        a.addWidget(b)
        b = QPushButton("Spec.")
        b.setObjectName("glopospec")
        b.setCheckable(True)
        self.globspec_bg.addButton(b)
        self.globspec_bg.setId(b, 1)
        a.addWidget(b)
        self.stack_globspec = QStackedLayout()
        self.globspec_bg.idClicked.connect(self.stack_globspec.setCurrentIndex)
        throw_layout.addLayout(a)
        self.globDet_layout = QVBoxLayout()
        self.globDet_layout.addWidget(QLabel("Alive battery threshold (%): "))
        self.globDet_layout.addWidget(QLabel("Alpha (metric param.): "))
        self.globDet_layout.addWidget(QLabel("Area rect height (D<sub>0</sub>): "))
        self.globDet_layout.addWidget(QLabel("Battery energy (E<sub>0</sub>): "))
        self.globDet_layout.addWidget(QLabel("b: path loss exp.: "))
        self.globDet_layout.addWidget(QLabel("Data origin probability: "))
        self.globDet_layout.addWidget(QLabel("Enhanced nodes (EN) rate (%): "))
        self.globDet_layout.addWidget(QLabel("Extra energy of ENs (%): "))
        self.globDet_layout.addWidget(QLabel("Frame length (B): "))
        self.globDet_layout.addWidget(QLabel("Iterations for meas. (% of popu.): "))
        self.globDet_layout.addWidget(QLabel("Number of sensors: "))
        self.globDet_layout.addWidget(QLabel("Wakeful probability: "))
        t = QWidget()
        t.setObjectName("globDetW")
        t.setLayout(self.globDet_layout)
        c = QScrollArea()
        c.setWidgetResizable(True)
        c.setWidget(t)
        self.specDet_layout = QVBoxLayout()
        t = QWidget()
        t.setObjectName("globDetW")
        t.setLayout(self.specDet_layout)
        e = QScrollArea()
        e.setWidgetResizable(True)
        e.setWidget(t)
        self.stack_globspec.addWidget(c)
        self.stack_globspec.addWidget(e)
        throw_layout.addLayout(self.stack_globspec)
        ana_para_layout.addLayout(throw_layout, 0, 2, 1, 1)
        #fourth column-------------------------------------------------------------------------
        # forow_layout = QVBoxLayout()
        # forow_layout.addWidget(QLabel("What to vis."))
        self.whattovisList = QListWidget()
        self.whattovisList.setObjectName("whatToW")
        n = QListWidgetItem("Number of frames received by the sink")
        self.whattovisList.addItem(n)
        n = QListWidgetItem("Average hopcount of all measurements")
        self.whattovisList.addItem(n)
        n = QListWidgetItem("Average of all failed meas. transmissions rate (%)")
        self.whattovisList.addItem(n)
        n = QListWidgetItem("Average charge rate at live nodes (%)")
        self.whattovisList.addItem(n)
        n = QListWidgetItem("Average number of frame forwarding")
        self.whattovisList.addItem(n)
        n = QListWidgetItem("Average energy for all comm. rate (%)")
        self.whattovisList.addItem(n)
        n = QListWidgetItem("All routing algorithm fails")
        self.whattovisList.addItem(n)
        n = QListWidgetItem("All frames received by the sink (bar chart)")
        self.whattovisList.addItem(n)
        n = QListWidgetItem("Average hopcount (bar chart)")
        self.whattovisList.addItem(n)
        n = QListWidgetItem("Lifetime in rounds (bar chart)")
        self.whattovisList.addItem(n)
        self.whattovisList.itemSelectionChanged.connect(self.whatToSelected)
        # forow_layout.addWidget(self.whattovisList)
        ana_para_layout.addWidget(self.whattovisList, 0, 3, 1, 1)
        # self.whattovisIsLine = [True, False, False, True, True, True, True, True, True]
        #fifth column-------------------------------------------------------------------------
        fifrow_layout = QVBoxLayout()
        fifrow_layout.addStretch()
        self.autoGrDr = QCheckBox("Live chart drawing")
        self.autoGrDr.setChecked(True)
        fifrow_layout.addWidget(self.autoGrDr)
        fifrow_layout.addWidget(QLabel("Curr. Sim. draw into:"))
        n = QRadioButton("New chart")
        n.setChecked(True)
        fifrow_layout.addWidget(n)
        self.drawinto_bg = QButtonGroup()
        self.drawinto_bg.addButton(n)
        self.drawinto_bg.setId(n, 0)
        n = QRadioButton("The existing chart")
        fifrow_layout.addWidget(n)
        self.drawinto_bg.addButton(n)
        self.drawinto_bg.setId(n, 1)
        self.isSensDataShown = False
        self.oriSensLocs = None
        self.prevNrun = None
        self.prevSug = 0
        self.prevCount = 0
        self.showDataButton = QPushButton("Show data of sensors")
        self.showDataButton.clicked.connect(self.showSensorData)
        fifrow_layout.addWidget(self.showDataButton)
        self.exportPngButton = QPushButton("Export graph to png")
        self.exportPngButton.clicked.connect(self.exportPng)
        fifrow_layout.addWidget(self.exportPngButton)
        ana_para_layout.addLayout(fifrow_layout, 0, 4, 1, 1)
        # -------------------------------------------------------------------------------------
        ana_para_layout.setColumnStretch(0, 10)
        ana_para_layout.setColumnStretch(1, 26)
        ana_para_layout.setColumnStretch(2, 26)
        ana_para_layout.setColumnStretch(3, 26)
        ana_para_layout.setColumnStretch(4, 12)
        wapl = QWidget()
        wapl.setLayout(ana_para_layout)
        # ha kell terulet: 1. ujabb stack: scroll eventtel valtas, 2. settingsben scrollArea or tabs
        # scwapl = QScrollArea()
        # scwapl.setObjectName("topscroll")
        # scwapl.setWidgetResizable(True)
        # scwapl.setWidget(wapl)
        self.top_para_layout.addWidget(wapl)
        
        self.qbg.idClicked.connect(self.change_para_set)
        
        # -------------------------------------------------------------------------------------
        self.runsPerModule = []
        for i in range(algNameListLen):
            self.refresh_modules()
        # -------------------------------------------------------------------------------------
        
        l_top_layout = QHBoxLayout()
        l_top_layout.addWidget(w_control_layout, 1, Qt.AlignTop)
        l_top_layout.addLayout(self.top_para_layout, 6)
        l_top_layout.setContentsMargins(0, 0, 0, 0)
        
        l_bottom_layout = QHBoxLayout()
        self.sens_draw_en = False
        self.sim_widget = Sim_widget()
        self.disp_combobox = QComboBox()
        self.disp_combobox.addItems(["Charge level", "Lifetime", "No. meas. received by sink", "Number of measurements", "Sent own measurements", "Forwarded measurements", "All sent measurements", "Total energy for all comm.", "Number of lost frames (own)", "Number of lost frames (fw)", "Number of routing alg. fails", "Average hop count", "Own meas. received by sink rate", "Successfull sending rate"])
        # self.disp_combobox.currentIndexChanged.connect(self.sim_widget.update)
        self.disp_combobox.currentIndexChanged.connect(self.sharedMinMax)
            
        skala_layout = QVBoxLayout()
        skala_layout.addWidget(self.disp_combobox)
        self.skala_widget = Skala_widget()
        # self.disp_combobox.currentIndexChanged.connect(self.skala_widget.update)
        skala_layout.addWidget(self.skala_widget)
        l_bottom_layout.addWidget(self.sim_widget, 9, Qt.AlignTop)
        l_bottom_layout.addLayout(skala_layout, 1)
        l_bottom_layout.setContentsMargins(0, 0, 0, 0)
        l_bottom_layout.setSpacing(0)        
        
        self.simSpace = QWidget()
        self.simSpace.setLayout(l_bottom_layout)
        self.l_bottom_stack = QStackedLayout()
        self.l_bottom_stack.addWidget(self.simSpace)
        self.chartView = mycallout.View()
        self.l_bottom_stack.addWidget(self.chartView)
        self.l_bottom_stack.setCurrentIndex(0)
        
        main_l_layout = QVBoxLayout()
        main_l_layout.addLayout(l_top_layout)
        main_l_layout.addLayout(self.l_bottom_stack)
        main_l_layout.setContentsMargins(0, 0, 0, 0)
        
        # main layout-------------------------------------------------
        main_layout = QHBoxLayout()
        main_layout.addLayout(main_l_layout, 6)
        main_layout.addLayout(main_r_layout, 1)
        main_layout.setContentsMargins(0, 0, 0, 0)        
        self.setLayout(main_layout)
        
        self.sensor_locs = []
        self.is_act_play = False
        self.natEnd = False
        # self.comm_pairs = []
        self.nrun = 0
        self.prevSelRun = 0
        self.globParaNames = ["Alive battery threshold (%): ", "Alpha (metric param.): ", "Area rect height (D<sub>0</sub>): ", "Battery energy (E<sub>0</sub>): ", "b: path loss exp.: ", "Data origin probability: ", "Enhanced nodes (EN) rate (%): ", "Extra energy of ENs (%): ", "Frame length (B): ", "Iterations for meas. (% of popu.): ", "Number of sensors: ", "Wakeful probability: "]
        self.mapSortedToGlob = [2, 4, 0, 1, 5, 10, 6, 9, 7, 11, 3, 8]
        # self.d0 = 87.5
        self.hatotav = 0
        self.sensd0 = 0
        self.liveOffset = 0
        self.roundSampleRate = 10
        self.grafColors = [220, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0, 255, 0, 255, 0, 255, 255, 127, 127, 0, 127, 0, 127, 0, 127, 127, 0, 0, 0]
        self.barFont = QFont("Helvetica", 20, QFont.Bold)
        self.barColor = QColor(9, 40, 0)
        self.isEnhSens = 0
        self.bCArray = 0
        self.oldOffsetCommPair = 0
        self.oldPnsd = 0
        random.seed()
    
    @Slot()
    def sharedMinMax(self):
        if self.is_act_play or self.natEnd or self.isSensDataShown:
            tmax = -99999
            tmin = 99999
            ci = self.disp_combobox.currentIndex()
            # Charge level
            if ci == 0:
                tmin = 0
                tmax = 100
            # Lifetime
            elif ci == 1:
                for i in range(self.nrun.numProc):
                    val = self.nrun.procData[6][i]
                    if val < tmin:
                        tmin = val
                    if val > tmax:
                        tmax = val
            # No. meas. received by sink
            elif ci == 2:
                for i in range(self.nrun.numProc):
                    val = self.nrun.procData[9][i]
                    if val < tmin:
                        tmin = val
                    if val > tmax:
                        tmax = val
            # Number of measurements
            elif ci == 3:
                for i in range(self.nrun.numProc):
                    val = self.nrun.procData[7][i]
                    if val < tmin:
                        tmin = val
                    if val > tmax:
                        tmax = val
            # Sent own measurements
            elif ci == 4:
                for i in range(self.nrun.numProc):
                    val = self.nrun.procData[0][i] + self.nrun.procData[1][i]
                    if val < tmin:
                        tmin = val
                    if val > tmax:
                        tmax = val
            # Forwarded measurements
            elif ci == 5:
                for i in range(self.nrun.numProc):
                    val = self.nrun.procData[2][i] + self.nrun.procData[3][i]
                    if val < tmin:
                        tmin = val
                    if val > tmax:
                        tmax = val
            # All sent measurements
            elif ci == 6:
                for i in range(self.nrun.numProc):
                    val = self.nrun.procData[0][i] + self.nrun.procData[1][i] + self.nrun.procData[2][i] + self.nrun.procData[3][i]
                    if val < tmin:
                        tmin = val
                    if val > tmax:
                        tmax = val
            # Total energy for all comm.
            elif ci == 7:
                tmin = 0
                tmax = 100
            # Number of lost frames (own)
            elif ci == 8:
                for i in range(self.nrun.numProc):
                    val = self.nrun.procData[4][i]
                    if val < tmin:
                        tmin = val
                    if val > tmax:
                        tmax = val
            # Number of lost frames (fw)
            elif ci == 9:
                for i in range(self.nrun.numProc):
                    val = self.nrun.procData[5][i]
                    if val < tmin:
                        tmin = val
                    if val > tmax:
                        tmax = val
            # Number of routing alg. fails
            elif ci == 10:
                for i in range(self.nrun.numProc):
                    val = self.nrun.procData[10][i]
                    if val < tmin:
                        tmin = val
                    if val > tmax:
                        tmax = val
            # Average hop count
            elif ci == 11:
                for i in range(self.nrun.numProc):
                    val = (0 if self.nrun.procData[9][i] == 0 else self.nrun.procData[11][i] / self.nrun.procData[9][i])
                    if val < tmin:
                        tmin = val
                    if val > tmax:
                        tmax = val
            # Own meas. received by sink rate
            elif ci == 12:
                for i in range(self.nrun.numProc):
                    nev = self.nrun.procData[0][i] + self.nrun.procData[1][i]
                    val = (0 if nev == 0 else 100 * self.nrun.procData[9][i] / nev)
                    if val < tmin:
                        tmin = val
                    if val > tmax:
                        tmax = val
            # Successfull sending rate
            elif ci == 13:
                for i in range(self.nrun.numProc):
                    partSum = self.nrun.procData[0][i] + self.nrun.procData[1][i] + self.nrun.procData[2][i] + self.nrun.procData[3][i]
                    nev = partSum + self.nrun.procData[4][i] + self.nrun.procData[5][i] + self.nrun.procData[10][i]
                    val = (0 if nev == 0 else 100 * partSum / nev)
                    if val < tmin:
                        tmin = val
                    if val > tmax:
                        tmax = val
            self.nrun.statMin = tmin
            self.nrun.statMax = tmax
            self.skala_widget.update()
            self.sim_widget.update()
        
    @Slot()
    def refresh_modules(self):
        mnames = [name for _, name, _ in pkgutil.iter_modules(['si_algs'])]
        for i in range(len(mnames)):
            jel = True
            for j in range(self.alg_count):
                if mnames[i] == self.routing_alg_names[j]:
                    jel = False
                    break
            if jel:
                # print("new module:", mnames[i])
                klass = getattr(importlib.import_module("si_algs." + mnames[i]), mnames[i])
                dex = len(self.routing_algs)
                self.routing_algs.append(klass())
                
                tspl = QVBoxLayout()
                iss = []
                refs = []
                for j in range(self.routing_algs[dex].para_count):
                    bw = QWidget()
                    bw.setObjectName("bw")
                    pala = QLabel(self.routing_algs[dex].para_names[j])
                    a = QVBoxLayout()
                    a.addWidget(pala)
                    if self.routing_algs[dex].para_types[j] == 2:
                        t = ComboBox(pala, pala.text(), self.routing_algs[dex].para_def_values[j])
                        t.addItems(self.routing_algs[dex].para_cat_values[j])
                        t.setCurrentIndex(self.routing_algs[dex].para_def_values[j])
                        a.addWidget(t)
                        refs.append(t)
                        iss.append(False)
                    elif self.routing_algs[dex].para_types[j] == 0:
                        t = Slider(0, self.routing_algs[dex].para_range_starts[j], self.routing_algs[dex].para_range_ends[j], self.routing_algs[dex].para_def_values[j], pala, pala.text())
                        a.addWidget(t)
                        refs.append(t)
                        iss.append(True)
                    elif self.routing_algs[dex].para_types[j] == 1:
                        t = Slider(1, self.routing_algs[dex].para_range_starts[j], self.routing_algs[dex].para_range_ends[j], self.routing_algs[dex].para_def_values[j], pala, pala.text())
                        a.addWidget(t)
                        refs.append(t)
                        iss.append(True)
                    bw.setLayout(a)
                    tspl.addWidget(bw)
                tspl.addStretch()
                tspl.setContentsMargins(0, 0, 0, 0)
                c = QWidget()
                c.setLayout(tspl)
                cqa = QScrollArea()
                cqa.setWidgetResizable(True)
                cqa.setWidget(c)
                self.spec_para_stacked_layout.insertWidget(self.alg_count, cqa)
                self.algs_spec_controls.append(refs)
                self.algs_spec_isSliders.append(iss)
                self.alg_count += 1
                self.routing_alg_names.append(mnames[i])
                    
                tmpli = QListWidgetItem(mnames[i])
                tmpli.setTextAlignment(Qt.AlignCenter)
                self.si_select_list.addItem(tmpli)
                
                m = QVBoxLayout()
                m.setContentsMargins(0, 0, 0, 0)
                m.addWidget(QLabel(mnames[i]))
                uu = QWidget()
                uu.setLayout(m)
                uu.hide()
                self.salrow_layout.addWidget(uu)
                self.laysPerModul.append(uu)
                self.runCountPerModul.append(0)
                self.runIdCountPerModul.append(0)
                self.runsPerModule.append([])
                
                z = QCheckBox(mnames[i])
                self.ana_alg_ch_bg.addButton(z)
                self.ana_alg_ch_bg.setId(z, self.alg_count - 1)
                self.falrow_layout.addWidget(z)
                
    
    @Slot()
    def set_to_default(self):
        x = self.si_select_list.currentRow()
        if not (x < 0 or x >= self.si_select_list.count()):
            for i in range(self.routing_algs[x].para_count):
                if self.algs_spec_isSliders[x][i]:
                    self.algs_spec_controls[x][i].ispaint = True
                    self.algs_spec_controls[x][i].setValue(self.algs_spec_controls[x][i].orislvalue)
                else:
                    self.algs_spec_controls[x][i].setCurrentIndex(self.algs_spec_controls[x][i].defindex)
    
    @Slot()
    def change_para_set(self, dex):
        self.top_para_layout.setCurrentIndex(dex)
    
    @Slot()
    def locate_sensors(self):
        numSens = self.glob_para_controls[self.NOSENSORS].sajat_val
        basicen = self.glob_para_controls[self.BATTEN].sajat_val
        self.isEnhSens = [False] * numSens
        largeen = basicen * (1 + self.glob_para_controls[self.EXTRAENERGY].sajat_val / 100)
        largeCount = int(numSens * self.glob_para_controls[self.ENHNODERATE].sajat_val / 100)
        self.bCArray = [basicen] * numSens
        for i in range(largeCount):
            x = int(random.random() * (numSens - i))
            j = -1
            k = 0
            while j != x:
                if self.bCArray[k] == basicen:
                    j += 1
                k += 1
            self.bCArray[k - 1] = largeen
            self.isEnhSens[k - 1] = True
        
        self.sens_draw_en = True
        self.sim_widget.update()
        if not self.play_button.isEnabled():
            self.play_button.setEnabled(True)
        
    @Slot()
    def do_play(self):
        x = self.si_select_list.currentRow()
        if x < 0 or x >= self.si_select_list.count():
            return
        # paras = []
        # for i in range(self.routing_algs[x].para_count):
        #     if self.algs_spec_isSliders[x][i]:
        #         paras.append(self.algs_spec_controls[x][i].sajat_val)
        #     else:
        #         paras.append(self.algs_spec_controls[x][i].currentText())
        # for i in range(len(self.glob_para_controls)):
        #     print(i, self.glob_para_controls[i].sajat_val)
        # print(paras)
        # print(self.speed_slider.sajat_val)
        # self.routing_algs[x].run(paras)
        if self.under_play:
            self.under_play = False
            self.remainder = self.play_timer.remainingTime()
            self.play_timer.stop()
            self.nrun.pause()
            self.play_button.setIcon(QApplication.style().standardIcon(QStyle.SP_MediaPlay))
        else:
            self.under_play = True
            if self.remainder == 0:
                self.play_timer.start(1000)
            else:
                self.play_timer.start(self.remainder)
            self.play_button.setIcon(QApplication.style().standardIcon(QStyle.SP_MediaPause))
            if not self.is_act_play:
                self.locsens_button.setEnabled(False)
                self.is_act_play = True
                self.natEnd = False
                # create simrun instance
                self.nrun = Sim_run(x, self.isGroupRunsCh.isChecked(), self.runIdCountPerModul[x], self)
                # fill with alg spec params
                for i in range(self.routing_algs[x].para_count):
                    if self.algs_spec_isSliders[x][i]:
                        self.nrun.spec_paras.append(self.algs_spec_controls[x][i].sajat_val)
                    else:
                        self.nrun.spec_paras.append(self.algs_spec_controls[x][i].currentText())
                # fill with global params
                # for i in range(len(self.glob_para_controls)):
                for i in range(self.glob_para_len):
                    self.nrun.glob_paras.append(self.glob_para_controls[i].sajat_val)
                # if batch run
                if self.isGroupRunsCh.isChecked():
                    tz = self.grouprun_bg.checkedId()
                    if tz == 0:
                        print(self.groupRunGlobCb.currentText())
                        print(self.glob_para_controls[self.mapSortedToGlob[self.groupRunGlobCb.currentIndex()]].sajat_val)
                    else:
                        print(self.groupRunSpecCb.currentText())
                        print(self.algs_spec_controls[self.si_select_list.currentRow()][self.groupRunSpecCb.currentIndex()].sajat_val)
                    print(self.fromSlider.sajat_val)
                    print(self.toSlider.sajat_val)
                    print(self.stepSlider.sajat_val)
                # start sim.
                self.nrun.begRun()
            else:
                self.nrun.contin()

            # nos = self.glob_para_controls[self.NOSENSORS].sajat_val
            # z = int(nos / 4)
            # self.comm_pairs = []
            # for i in range(z):
            #     dex = int(random.random() * nos)
            #     k = dex
            #     while k == dex:
            #         k = int(random.random() * nos)
            #     self.comm_pairs.extend([dex, k])
            # self.sim_widget.update()
                
    
    @Slot()
    def do_stop(self, fend=False):
        self.play_timer.stop()
        self.playtime_count = 0
        self.remainder = 0
        self.playtime_label.setText("00:00")
        self.locsens_button.setEnabled(True)
        if fend:
            self.natEnd = True
            self.viewChart()
        if self.is_act_play:
            self.nrun.stop()
            x = self.nrun.moduleid
            self.runsPerModule[x].append(self.nrun)
            self.runCountPerModul[x] += 1
            self.runIdCountPerModul[x] += 1
            tmpstr = ""
            if self.isGroupRunsCh.isChecked():
                tmpstr = "G_"
            nev = tmpstr + self.routing_alg_names[x] + "_" + str(self.runIdCountPerModul[x])
            self.nrun.renameSeries(nev)
            a = CheckBox(nev, x, self.runIdCountPerModul[x] - 1, self, self.nrun)
            self.laysPerModul[x].layout().addWidget(a)
        self.is_act_play = False
        if not fend:
            self.nrun = 0
            self.natEnd = False
        if self.under_play:
            self.under_play = False
            self.play_button.setIcon(QApplication.style().standardIcon(QStyle.SP_MediaPlay))
        self.sim_widget.update()
    
    @Slot()
    def inc_speed(self):
        self.speed_slider.setValue(self.speed_slider.value() + self.speed_nov)
    
    @Slot()
    def dec_speed(self):
        self.speed_slider.setValue(self.speed_slider.value() - self.speed_nov)
    
    @Slot()
    def time_update(self):
        if self.remainder != 0:
            self.play_timer.start(1000)
            self.remainder = 0
        self.playtime_count += 1
        p = ""
        m = ""
        perc = self.playtime_count // 60
        mp = self.playtime_count % 60
        if perc // 10 == 0:
            p = "0"
        if mp // 10 == 0:
            m = "0"
        self.playtime_label.setText(p + str(perc) + ":" + m + str(mp))
        
    @Slot()
    def commrChanged(self, value):
        if not self.is_act_play:
            if self.sim_widget.arc_idx == -1:
                self.sim_widget.arc_idx = -2
            self.sim_widget.update()
    
    @Slot()
    def listRunsOfModul(self, idx, isch):
        if isch:
            self.laysPerModul[idx].setVisible(True)
            # self.salrow_layout.addLayout(self.laysPerModul[idx])
        else:
            self.laysPerModul[idx].hide()
            # self.salrow_layout.removeItem(self.laysPerModul[idx])
            
    @Slot()
    def selectAll(self):
        for i in range(self.alg_count):
            for j in range(1, self.runCountPerModul[i] + 1):
                self.laysPerModul[i].layout().itemAt(j).widget().setCheckState(Qt.Checked)

    @Slot()
    def deSelectAll(self):
        for i in range(self.alg_count):
            for j in range(1, self.runCountPerModul[i] + 1):
                self.laysPerModul[i].layout().itemAt(j).widget().setCheckState(Qt.Unchecked)
    
    @Slot()
    def listParasOfRun(self, modid, actrun):
        # actrun = 0
        # for i in range(len(self.runsPerModule[modid])):
        #     if self.runsPerModule[modid][i].runid == runid:
        #         actrun = self.runsPerModule[modid][i]
        #         break
        for i in range(len(self.glob_para_controls)):
            vstr = ""
            if self.glob_para_controls[self.mapSortedToGlob[i]].ptype == 0:
                vstr = str(actrun.glob_paras[self.mapSortedToGlob[i]])
            else:
                vstr = '{:.2f}'.format(actrun.glob_paras[self.mapSortedToGlob[i]])
            self.globDet_layout.itemAt(i).widget().setText(self.globParaNames[i] + "<b>" + vstr + "</b>")
        for i in range(self.specDet_layout.count() - 1, -1, -1):
            ff = self.specDet_layout.itemAt(i).widget()
            self.specDet_layout.removeWidget(ff)
            ff.hide()
        for i in range(self.routing_algs[modid].para_count):
            self.specDet_layout.addWidget(QLabel(self.routing_algs[modid].para_names[i] + ": <b>" + str(actrun.spec_paras[i]) + "</b>"))
            
    @Slot()
    def loadRun(self):
        files, k = QFileDialog.getOpenFileNames(self,
                "Load run file", ".",
                "wnsi files (*.wnsi)", "", QFileDialog.Options())
        if files:
            for fname in files:
                with open(fname, 'rb') as f:
                    arun = pickle.load(f)
                    tgs = [QtCharts.QLineSeries() for i in range(7)]
                    for i in range(7):
                        tgs[i].append(arun.grafSeries[i])
                        gpen = tgs[i].pen()
                        gpen.setWidth(5)
                        tgs[i].setPen(gpen)
                    arun.grafSeries = tgs
                    moname = arun.modulename
                    x = 0
                    for i in range(len(self.routing_alg_names)):
                        if self.routing_alg_names[i] == moname:
                            x = i
                            break
                    self.runsPerModule[x].append(arun)
                    self.runCountPerModul[x] += 1
                    self.runIdCountPerModul[x] += 1
                    nev = fname.split('/')[-1][:-5]
                    arun.renameSeries(nev)
                    a = CheckBox(nev, x, self.runIdCountPerModul[x] - 1, self, arun)
                    self.laysPerModul[x].layout().addWidget(a)
                
    @Slot()
    def saveRun(self):
        dirName = QFileDialog.getExistingDirectory(self,
                "Save runs into an existing directory", ".", QFileDialog.ShowDirsOnly)
        if dirName:
            for i in range(self.alg_count):
                for j in range(self.runCountPerModul[i], 0, -1):
                    ff = self.laysPerModul[i].layout().itemAt(j).widget()
                    if ff.isChecked():
                        with open(dirName + '/' + ff.text() + ".wnsi", 'wb') as f:
                            pickle.dump(ff.szimrun, f)
    
    @Slot()
    def delChoosed(self):
        for i in range(self.alg_count):
            for j in range(self.runCountPerModul[i], 0, -1):
                ff = self.laysPerModul[i].layout().itemAt(j).widget()
                if ff.isChecked():
                    self.runCountPerModul[i] -= 1
                    self.laysPerModul[i].layout().removeWidget(ff)
                    ff.hide()
                    self.runsPerModule[i].remove(ff.szimrun)
                    # runid = ff.runid
                    # for k in range(len(self.runsPerModule[i])):
                    #     if self.runsPerModule[i][k].runid == runid:
                    #         self.runsPerModule[i].pop(k)
                    #         break
        
    @Slot()
    def viewChart(self):
        # print("vislist:", self.whattovisList.currentRow())
        # print("isgroup", self.whattovisIsLine[self.whattovisList.currentRow()])
        # print("autoGrDr", self.autoGrDr.isChecked())
        # print("drawinto", self.drawinto_bg.checkedId())
        # for i in range(self.alg_count):
        #     for j in range(1, self.runCountPerModul[i] + 1):
        #         ff = self.laysPerModul[i].layout().itemAt(j).widget()
        #         if ff.isChecked():
        #             seriesList.append(ff.szimrun.grafSeries[cr])
                    # runid = ff.runid
                    # actrun = 0
                    # for k in range(len(self.runsPerModule[i])):
                    #     if self.runsPerModule[i][k].runid == runid:
                    #         actrun = self.runsPerModule[i][k]
                    #         break
                    # print(actrun.glob_paras)
                    # print(actrun.spec_paras)
        
        cr = self.whattovisList.currentRow()
        noDraw = False
        if cr < 0 or cr >= self.whattovisList.count():
            if self.natEnd:
                noDraw = True
            else:
                return
        
        if self.is_act_play or self.natEnd:
            if self.liveOffset != -1:
                toOff = 9999999
                for i in range(self.nrun.numProc):
                    if self.nrun.battCapArray[i] != -1:
                        za = int(self.nrun.procData[6][i] / self.roundSampleRate) - 1
                        if za < toOff:
                            toOff = za
                if toOff > self.liveOffset and toOff != 9999999:
                    for j in [0, 6]:
                        self.nrun.grafSeries[j].append([QPointF(i + 1, self.nrun.grafShMem[j][i]) for i in range(self.liveOffset, toOff)])
                    cc = self.nrun.numProc
                    for j in [1, 4]:
                        self.nrun.grafSeries[j].append([QPointF(i + 1, self.nrun.grafShMem[j][i] / cc) for i in range(self.liveOffset, toOff)])
                    for j in [2, 5]:
                        self.nrun.grafSeries[j].append([QPointF(i + 1, 100 * self.nrun.grafShMem[j][i] / cc) for i in range(self.liveOffset, toOff)])
                    self.nrun.grafSeries[3].append([QPointF(i + 1, 100 * self.nrun.grafShMem[3][i] / self.nrun.grafShMem[7][i]) for i in range(self.liveOffset, toOff)])
                    self.liveOffset = toOff
                if self.natEnd and toOff != 9999999:
                    for j in [0, 6]:
                        self.nrun.grafSeries[j].append(QPointF(toOff + 1, self.nrun.grafShMem[j][toOff]))
                    for j in [1, 4]:
                        self.nrun.grafSeries[j].append(QPointF(toOff + 1, self.nrun.grafShMem[j][toOff] / self.nrun.numProc))
                    for j in [2, 5]:
                        self.nrun.grafSeries[j].append(QPointF(toOff + 1, 100 * self.nrun.grafShMem[j][toOff] / self.nrun.numProc))
                    self.nrun.grafSeries[3].append(QPointF(toOff + 1, 100 * self.nrun.grafShMem[3][toOff] / self.nrun.grafShMem[7][toOff]))
                    self.liveOffset = -1
        
        if noDraw:
            return
        
        self.chartView.clearSeri()
        isLine = cr < 7
        
        if isLine:
            seriesList = []
            if self.is_act_play or self.natEnd:
                seriesList.append(self.nrun.grafSeries[cr])
            for i in range(self.alg_count):
                for j in range(1, self.runCountPerModul[i] + 1):
                    ff = self.laysPerModul[i].layout().itemAt(j).widget()
                    if ff.isChecked():
                        seriesList.append(ff.szimrun.grafSeries[cr])
            for i, seri in zip(range(len(seriesList)), seriesList):
                bs = seri.pen()
                dd = (i % 10) * 3
                bs.setColor(QColor(self.grafColors[dd], self.grafColors[dd + 1], self.grafColors[dd + 2]))
                seri.setPen(bs)
            self.chartView.setSeries(seriesList)
            self.chartView.setXTitle("rounds [10x]")
            fmat = ""
            if cr == 0 or cr == 6:
                fmat = '%.0f'
            elif cr != 1:
                fmat = '%.1f'
            else:
                fmat = '%.2f'
            self.chartView.setYFormat(fmat)
            # if not self.is_act_play:
            #     self.chartView.setXNice()
            #     if cr == 0 or cr == 4 or cr == 6:
            #         self.chartView.setYNice()
        else:
            barValues = []
            cats = []
            for i in range(self.alg_count):
                for j in range(1, self.runCountPerModul[i] + 1):
                    ff = self.laysPerModul[i].layout().itemAt(j).widget()
                    if ff.isChecked():
                        cats.append(ff.text())
                        if cr == 7:
                            barValues.append(ff.szimrun.sens_datas[3])
                        elif cr == 8:
                            avghop = 0
                            for oo in range(ff.szimrun.numProc):
                                avghop += (0 if ff.szimrun.procData[9][oo] == 0 else ff.szimrun.procData[11][oo] / ff.szimrun.procData[9][oo])
                            avghop /= ff.szimrun.numProc
                            barValues.append(avghop)
                        else:
                            maxRound = 0
                            for oo in range(ff.szimrun.numProc):
                                zk = ff.szimrun.procData[6][oo]
                                if zk > maxRound:
                                    maxRound = zk
                            barValues.append(maxRound)
            if len(barValues) == 0:
                return
            bsname = ""
            if cr == 7:
                bsname = "All frames received by the sink [ ]"
            elif cr == 8:
                bsname = "Average hopcount [ ]"
            else:
                bsname = "Lifetime in rounds [ ]"
            barset = QtCharts.QBarSet(bsname)
            barset.setLabelFont(self.barFont)
            barset.setLabelColor(Qt.white)
            barset.setColor(self.barColor)
            barset.append(barValues)
            barSeries = QtCharts.QBarSeries()
            barSeries.setLabelsPosition(QtCharts.QAbstractBarSeries.LabelsInsideEnd)
            barSeries.setLabelsVisible()
            barSeries.append(barset)
            self.chartView.setBarSeries(barSeries, cats, max(barValues))
            fmat = ""
            if cr == 7 or cr == 9:
                fmat = '%.0f'
            else:
                fmat = '%.2f'
            self.chartView.setYFormat(fmat)
            # if not self.is_act_play and (cr == 7 or cr == 9):
            #     self.chartView.setYNice()
        
        if isLine:
            if cr == 0:
                self.chartView.setYTitle("No. frames received by the sink [ ]")
            elif cr == 1:
                self.chartView.setYTitle("Average hopcount of all measurements [ ]")
            elif cr == 2:
                self.chartView.setYTitle("Average of all failed meas. transmissions rate [%]")
            elif cr == 3:
                self.chartView.setYTitle("Average charge rate at live nodes [%]")
            elif cr == 4:
                self.chartView.setYTitle("Average number of frame forwarding [ ]")
            elif cr == 5:
                self.chartView.setYTitle("Average energy for all comm. rate [%]")
            elif cr == 6:
                self.chartView.setYTitle("All routing algorithm fails [ ]")
        
        if self.l_bottom_stack.currentIndex() != 1:
            self.l_bottom_stack.setCurrentIndex(1)
        
    @Slot()
    def groupRuns(self, isch):
        if isch:
            self.specOrGlobBW.setVisible(True)
            self.specOrGlobStW.setVisible(True)
        else:
            self.specOrGlobBW.hide()
            self.specOrGlobStW.hide()
            self.ftoSl_layout_W.hide()
            self.fto_step_layout_W.hide()
    
    @Slot()
    def showGlobFromToSliders(self, dex):
        x = self.glob_para_controls[self.mapSortedToGlob[dex]].ptype
        if x != 2:
            self.ftoSl_layout.removeWidget(self.fromSlider)
            self.fromSlider.hide()
            self.ftoSl_layout.removeWidget(self.toSlider)
            self.toSlider.hide()
            self.ftoSl_layout.removeWidget(self.stepSlider)
            self.stepSlider.hide()
            asc = self.glob_para_controls[self.mapSortedToGlob[dex]]
            if x == 0:
                self.fromSlider = Slider(0, asc.sajat_min, asc.sajat_max, asc.sajat_min, self.fromSlLabel, "From")
                self.toSlider = Slider(0, asc.sajat_min, asc.sajat_max, asc.sajat_max, self.toSlLabel, "To")
                ks = asc.sajat_max - asc.sajat_min
                self.stepSlider = Slider(0, 0, ks, int(ks / 2), self.stepSlLabel, "Step")
            else:
                self.fromSlider = Slider(1, asc.sajat_min, asc.sajat_max, asc.sajat_min, self.fromSlLabel, "From")
                self.toSlider = Slider(1, asc.sajat_min, asc.sajat_max, asc.sajat_max, self.toSlLabel, "To")
                ks = asc.sajat_max - asc.sajat_min
                self.stepSlider = Slider(1, 0, ks, ks / 2, self.stepSlLabel, "Step")
            self.ftoSl_layout.insertWidget(1, self.fromSlider)
            self.ftoSl_layout.insertWidget(3, self.toSlider)
            self.fto_step_layout.insertWidget(1, self.stepSlider)
            self.ftoSl_layout_W.setVisible(True)
            self.fto_step_layout_W.setVisible(True)
        else:
            self.ftoSl_layout_W.hide()
            self.fto_step_layout_W.hide()
    
    @Slot()
    def showSpecFromToSliders(self, dex):
        if self.isGroupRunsCh.isChecked() and self.grouprun_bg.checkedId() == 1:
            cur = self.si_select_list.currentRow()
            x = self.routing_algs[cur].para_types[dex]
            if x != 2:
                self.ftoSl_layout.removeWidget(self.fromSlider)
                self.fromSlider.hide()
                self.ftoSl_layout.removeWidget(self.toSlider)
                self.toSlider.hide()
                self.ftoSl_layout.removeWidget(self.stepSlider)
                self.stepSlider.hide()
                asc = self.algs_spec_controls[cur][dex]
                if x == 0:
                    self.fromSlider = Slider(0, asc.sajat_min, asc.sajat_max, asc.sajat_min, self.fromSlLabel, "From")
                    self.toSlider = Slider(0, asc.sajat_min, asc.sajat_max, asc.sajat_max, self.toSlLabel, "To")
                    ks = asc.sajat_max - asc.sajat_min
                    self.stepSlider = Slider(0, 0, ks, int(ks / 2), self.stepSlLabel, "Step")
                else:
                    self.fromSlider = Slider(1, asc.sajat_min, asc.sajat_max, asc.sajat_min, self.fromSlLabel, "From")
                    self.toSlider = Slider(1, asc.sajat_min, asc.sajat_max, asc.sajat_max, self.toSlLabel, "To")
                    ks = asc.sajat_max - asc.sajat_min
                    self.stepSlider = Slider(1, 0, ks, ks / 2, self.stepSlLabel, "Step")
                self.ftoSl_layout.insertWidget(1, self.fromSlider)
                self.ftoSl_layout.insertWidget(3, self.toSlider)
                self.fto_step_layout.insertWidget(1, self.stepSlider)
                self.ftoSl_layout_W.setVisible(True)
                self.fto_step_layout_W.setVisible(True)
            else:
                self.ftoSl_layout_W.hide()
                self.fto_step_layout_W.hide()
    
    @Slot()
    def uploadSpecCb(self, dex):
        if not (dex < 0 or dex >= self.si_select_list.count()):
            self.groupRunSpecCb.clear()
            for i in range(self.routing_algs[dex].para_count):
                self.groupRunSpecCb.addItem(self.routing_algs[dex].para_names[i])
    
    @Slot()
    def switchToSimView(self):
        if self.l_bottom_stack.currentIndex() == 1:
            self.l_bottom_stack.setCurrentIndex(0)
    
    @Slot()
    def showSensorData(self):
        if self.isSensDataShown:
            self.sensor_locs = self.oriSensLocs
            self.nrun = self.prevNrun
            self.sim_widget.sens_sug = self.prevSug
            self.sim_widget.sens_count = self.prevCount
            self.isSensDataShown = False
            self.showDataButton.setText("Show data of sensors")
            if self.is_act_play or self.natEnd:
                self.sharedMinMax()
            else:
                self.sim_widget.update()
                self.skala_widget.update()
        elif self.sim_widget.volt_elso:
            jel = False
            for i in range(self.alg_count):
                # for j in range(self.runCountPerModul[i], 0, -1):
                for j in range(1, self.runCountPerModul[i] + 1):
                    ff = self.laysPerModul[i].layout().itemAt(j).widget()
                    if ff.isChecked():
                        self.oriSensLocs = self.sensor_locs
                        self.prevNrun = self.nrun
                        self.prevSug = self.sim_widget.sens_sug
                        self.prevCount = self.sim_widget.sens_count
                        self.sensor_locs = ff.szimrun.locOfSens
                        self.nrun = ff.szimrun
                        self.sim_widget.sens_sug = ff.szimrun.sensSug
                        self.sim_widget.sens_count = ff.szimrun.numProc
                        self.isSensDataShown = True
                        self.showDataButton.setText("Hide data of sensors")
                        self.hatotav = ff.szimrun.hatotav
                        self.sharedMinMax()
                        jel = True
                        break
                if jel:
                    break
    
    @Slot()
    def exportPng(self):
        fileName, k = QFileDialog.getSaveFileName(self,
                "Export graph to PNG", ".",
                "png files (*.png)", "", QFileDialog.Options())
        if fileName:
            file = QFile(fileName)
            file.open(QIODevice.WriteOnly)
            if self.l_bottom_stack.currentIndex() == 0:
                self.simSpace.grab().save(file, "PNG")
            else:
                self.chartView.grab().save(file, "PNG")
    
    @Slot()
    def whatToSelected(self):
        if self.l_bottom_stack.currentIndex() == 1:
            self.viewChart()



if __name__ == "__main__":    
    routing_alg_nevek_count = len([name for _, name, _ in pkgutil.iter_modules(['si_algs'])])
    if routing_alg_nevek_count == 0:
        sys.exit('Error: No module has been found in the si_algs package, or si_algs package is missing.')

    app = QApplication()
    
    w = Widget(routing_alg_nevek_count)
    availableGeometry = app.desktop().availableGeometry(w)
    # print(availableGeometry.width(), availableGeometry.height())
    w.resize(availableGeometry.width(), availableGeometry.height())
    w.showMaximized()

    with open("style.qss", "r") as f:
        _style = f.read()
        app.setStyleSheet(_style)

    sys.exit(app.exec_())