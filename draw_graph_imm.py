#!/usr/bin/env python3
import random
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import QTimer
import numpy as np
import pyqtgraph as pg
import sys

class MyWidget(QWidget):

    def __init__(self):
        super().__init__()
        self.setGeometry(400,200,500,500)
        pg.setConfigOptions(antialias=True)
        pg.setConfigOptions(foreground='k')
        pg.setConfigOptions(background='w')

        win = pg.GraphicsLayoutWidget(size=(400,300),border=True,parent=self)
        win.move(10,50)

        graph = win.addPlot(title="Data")
        graph.setLabel('left',"Power", units='W')
        graph.setLabel('bottom',"Time", units='s')

        self.counter = 0
        self.x = []
        self.y = []
        self.curve = graph.plot(self.x,self.y,pen=pg.mkPen((120,23,200),width=2))
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.add_data)
        self.timer.setSingleShot(True)
        self.add_data()
    
    def add_data(self):
        self.x.append(self.counter)
        self.y.append(random.random())
        self.counter += 1
        self.curve.setData(self.x, self.y)
        self.timer.start(500)

    
maxX = 100
app = QApplication(sys.argv)
gui = MyWidget()


gui.show()
sys.exit(app.exec_())  