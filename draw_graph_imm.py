#!/usr/bin/env python3
from PyQt5.QtWidgets import QApplication, QWidget
from pyqtgraph.Qt import QtGui, QtCore
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


maxX = 100
app = QApplication(sys.argv)
gui = MyWidget()


gui.show()
sys.exit(app.exec_())  