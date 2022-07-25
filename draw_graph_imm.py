#!/usr/bin/env python3
from PyQt5.QtWidgets import QApplication, QWidget
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import sys

maxX = 100
app = QApplication(sys.argv)
gui = QWidget() 
gui.setGeometry(400,200,500,500)

pg.setConfigOptions(antialias=True)
pg.setConfigOptions(foreground='k')
pg.setConfigOptions(background='w')
win = pg.GraphicsLayoutWidget(size=(400,300),border=True,parent=gui)
win.move(10,50)

gui.show()
sys.exit(app.exec_())  