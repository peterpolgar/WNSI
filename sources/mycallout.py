
#############################################################################
##
## Copyright (C) 2018 The Qt Company Ltd.
## Contact: http://www.qt.io/licensing/
##
## This file is part of the Qt for Python examples of the Qt Toolkit.
##
## $QT_BEGIN_LICENSE:BSD$
## You may use this file under the terms of the BSD license as follows:
##
## "Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are
## met:
##   * Redistributions of source code must retain the above copyright
##     notice, this list of conditions and the following disclaimer.
##   * Redistributions in binary form must reproduce the above copyright
##     notice, this list of conditions and the following disclaimer in
##     the documentation and/or other materials provided with the
##     distribution.
##   * Neither the name of The Qt Company Ltd nor the names of its
##     contributors may be used to endorse or promote products derived
##     from this software without specific prior written permission.
##
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
## "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
## LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
## A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
## OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
## SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
## LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
## DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
## THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
## (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
##
## $QT_END_LICENSE$
##
#############################################################################

"""PySide2 port of the Callout example from Qt v5.x"""

import sys
from PySide2.QtWidgets import (QApplication, QGraphicsScene,
    QGraphicsView, QGraphicsSimpleTextItem, QGraphicsItem, QSizePolicy)
from PySide2.QtCore import Qt, QPointF, QRectF, QRect
from PySide2.QtCharts import QtCharts
from PySide2.QtGui import QPainter, QFont, QFontMetrics, QPainterPath, QColor, QPen


class Callout(QGraphicsItem):

    def __init__(self, chart, par):
        QGraphicsItem.__init__(self, chart)
        self._chart = chart
        self._text = ""
        self._textRect = QRectF()
        self._anchor = QPointF()
        self._font = QFont()
        self._rect = QRectF()
        self.par = par

    def boundingRect(self):
        anchor = self.mapFromParent(self._chart.mapToPosition(self._anchor))
        rect = QRectF()
        rect.setLeft(min(self._rect.left(), anchor.x()))
        rect.setRight(max(self._rect.right(), anchor.x()))
        rect.setTop(min(self._rect.top(), anchor.y()))
        rect.setBottom(max(self._rect.bottom(), anchor.y()))

        return rect

    def paint(self, painter, option, widget):
        path = QPainterPath()
        path.addRoundedRect(self._rect, 5, 5)
        anchor = self.mapFromParent(self._chart.mapToPosition(self._anchor))
        if not self._rect.contains(anchor) and not self._anchor.isNull():
            point1 = QPointF()
            point2 = QPointF()

            # establish the position of the anchor point in relation to _rect
            above = anchor.y() <= self._rect.top()
            aboveCenter = (anchor.y() > self._rect.top() and
                anchor.y() <= self._rect.center().y())
            belowCenter = (anchor.y() > self._rect.center().y() and
                anchor.y() <= self._rect.bottom())
            below = anchor.y() > self._rect.bottom()

            onLeft = anchor.x() <= self._rect.left()
            leftOfCenter = (anchor.x() > self._rect.left() and
                anchor.x() <= self._rect.center().x())
            rightOfCenter = (anchor.x() > self._rect.center().x() and
                anchor.x() <= self._rect.right())
            onRight = anchor.x() > self._rect.right()

            # get the nearest _rect corner.
            x = (onRight + rightOfCenter) * self._rect.width()
            y = (below + belowCenter) * self._rect.height()
            cornerCase = ((above and onLeft) or (above and onRight) or
                (below and onLeft) or (below and onRight))
            vertical = abs(anchor.x() - x) > abs(anchor.y() - y)

            x1 = (x + leftOfCenter * 10 - rightOfCenter * 20 + cornerCase *
                int(not vertical) * (onLeft * 10 - onRight * 20))
            y1 = (y + aboveCenter * 10 - belowCenter * 20 + cornerCase *
                vertical * (above * 10 - below * 20))
            point1.setX(x1)
            point1.setY(y1)

            x2 = (x + leftOfCenter * 20 - rightOfCenter * 10 + cornerCase *
                int(not vertical) * (onLeft * 20 - onRight * 10))
            y2 = (y + aboveCenter * 20 - belowCenter * 10 + cornerCase *
                vertical * (above * 20 - below * 10))
            point2.setX(x2)
            point2.setY(y2)

            path.moveTo(point1)
            path.lineTo(anchor)
            path.lineTo(point2)
            path = path.simplified()

        painter.setBrush(QColor(255, 255, 255))
        painter.drawPath(path)
        painter.drawText(self._textRect, self._text)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.par.removeTooltip(self)
        event.setAccepted(True)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.setPos(self.mapToParent(
                event.pos() - event.buttonDownPos(Qt.LeftButton)))
            event.setAccepted(True)
        else:
            event.setAccepted(False)

    def setText(self, text):
        self._text = text
        metrics = QFontMetrics(self._font)
        self._textRect = QRectF(metrics.boundingRect(
            QRect(0.0, 0.0, 150.0, 150.0),Qt.AlignLeft, self._text))
        self._textRect.translate(5, 5)
        self.prepareGeometryChange()
        self._rect = self._textRect.adjusted(-5, -5, 5, 5)

    def setAnchor(self, point):
        self._anchor = QPointF(point)

    def updateGeometry(self):
        self.prepareGeometryChange()
        self.setPos(self._chart.mapToPosition(
            self._anchor) + QPointF(10, -50))


class View(QGraphicsView):
    def __init__(self, parent = None):
        super(View, self).__init__(parent)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.setScene(QGraphicsScene(self))

        self.setDragMode(QGraphicsView.NoDrag)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Chart
        self._chart = QtCharts.QChart()
        self._chart.setMinimumSize(640, 480)
        self._chart.setTitle("Hover the line to show callout. Click the line "
            "to make it stay. Right click on callout to remove it.")

        self._chart.setAcceptHoverEvents(True)

        self.setRenderHint(QPainter.Antialiasing)
        self.scene().addItem(self._chart)

        self._coordX = QGraphicsSimpleTextItem(self._chart)
        self._coordX.setPos(
            self._chart.size().width()/2 - 50, self._chart.size().height())
        self._coordX.setText("X: ")
        self._coordY = QGraphicsSimpleTextItem(self._chart)
        self._coordY.setPos(
            self._chart.size().width()/2 + 50, self._chart.size().height())
        self._coordY.setText("Y: ")

        self._callouts = []
        self._tooltip = Callout(self._chart, self)

        self.setMouseTracking(True)
        
        self.axisLabelFont = QFont("Helvetica", 12, QFont.Bold)
        self.axisTitleFont = QFont("Helvetica", 14, QFont.Bold)
        self.legendFont = self.axisLabelFont
        self.gridPen = QPen(Qt.darkGray)
        self.gridPen.setWidth(2)
        
        leg = self._chart.legend()
        leg.setFont(self.legendFont)

    def resizeEvent(self, event):
        if self.scene():
            self.scene().setSceneRect(QRectF(QPointF(0, 0), event.size()))
            self._chart.resize(event.size())
            self._coordX.setPos(
                self._chart.size().width()/2 - 50,
                self._chart.size().height() - 20)
            self._coordY.setPos(
                self._chart.size().width()/2 + 50,
                self._chart.size().height() - 20)
            for callout in self._callouts:
                callout.updateGeometry()
        QGraphicsView.resizeEvent(self, event)


    def mouseMoveEvent(self, event):
        self._coordX.setText("X: {0:.2f}"
            .format(self._chart.mapToValue(event.pos()).x()))
        self._coordY.setText("Y: {0:.2f}"
            .format(self._chart.mapToValue(event.pos()).y()))
        QGraphicsView.mouseMoveEvent(self, event)

    def keepCallout(self):
        self._callouts.append(self._tooltip)
        self._tooltip = Callout(self._chart, self)

    def tooltip(self, point, state):
        if self._tooltip == 0:
            self._tooltip = Callout(self._chart, self)

        if state:
            self._tooltip.setText("X: {0:.2f} \nY: {1:.2f} "
                .format(point.x(),point.y()))
            self._tooltip.setAnchor(point)
            self._tooltip.setZValue(11)
            self._tooltip.updateGeometry()
            self._tooltip.show()
        else:
            self._tooltip.hide()
            
    def removeTooltip(self, tt):
        tt.hide()
        self._callouts.remove(tt)
    
    def setSeries(self, mseries):
        smax = -999999
        smin = 999999
        for series in mseries:
            self._chart.addSeries(series)
            series.clicked.connect(self.keepCallout)
            series.hovered.connect(self.tooltip)
            sepo = series.points()
            if sepo:
                for gg in sepo:
                    kf = gg.y()
                    if kf > smax:
                        smax = kf
                    if kf < smin:
                        smin = kf
        if smin == 999999:
            smin = 0
            smax = 1
        elif smin < 0.1:
            smin = -smax * 0.05
        else:
            smin = 0
        self._chart.createDefaultAxes()
        xaxis = self._chart.axes(Qt.Horizontal)[0]
        # xaxis = QtCharts.QValueAxis()
        xaxis.setLabelFormat("%d")
        xaxis.setLabelsFont(self.axisLabelFont)
        xaxis.setTitleFont(self.axisTitleFont)
        xaxis.setGridLinePen(self.gridPen)
        # self._chart.addAxis(xaxis, Qt.AlignBottom)
        yaxis = self._chart.axes(Qt.Vertical)[0]
        # yaxis = QtCharts.QValueAxis()
        yaxis.setRange(smin, smax * 1.05)
        yaxis.setLabelsFont(self.axisLabelFont)
        yaxis.setTitleFont(self.axisTitleFont)
        yaxis.setGridLinePen(self.gridPen)
        # self._chart.addAxis(yaxis, Qt.AlignLeft)
    
    def setBarSeries(self, mseries, categories, smax):
        for ax in self._chart.axes(Qt.Horizontal):
            self._chart.removeAxis(ax)
        for ax in self._chart.axes(Qt.Vertical):
            self._chart.removeAxis(ax)
        self._chart.addSeries(mseries)
        axisX = QtCharts.QBarCategoryAxis()
        axisX.setLabelsFont(self.axisLabelFont)
        axisX.setTitleFont(self.axisTitleFont)
        axisX.append(categories)
        # axisX.setTitleText("Runs")
        self._chart.addAxis(axisX, Qt.AlignBottom)
        mseries.attachAxis(axisX)
        axisY = QtCharts.QValueAxis()
        axisY.setRange(0, smax * 1.05)
        axisY.setLabelsFont(self.axisLabelFont)
        # axisY.setTitleFont(self.axisTitleFont)
        self._chart.addAxis(axisY, Qt.AlignLeft)
        mseries.attachAxis(axisY)
    
    # def clearSeries(self):
    #     self._chart.removeAllSeries()
    
    def clearSeri(self):
        for seri in self._chart.series():
            self._chart.removeSeries(seri)
    
    def setXTitle(self, xtitle):
        self._chart.axes(Qt.Horizontal)[0].setTitleText(xtitle)
    
    def setYTitle(self, ytitle):
        self._chart.axes(Qt.Vertical)[0].setTitleText(ytitle)
    
    def setYFormat(self, fmat):
        self._chart.axes(Qt.Vertical)[0].setLabelFormat(fmat)

    def setXNice(self):
        self._chart.axes(Qt.Horizontal)[0].applyNiceNumbers()
    
    def setYNice(self):
        self._chart.axes(Qt.Vertical)[0].applyNiceNumbers()
