#!/usr/bin/env python3
import os
import random
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QFileDialog, QAction
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QIcon
import numpy as np
import pyqtgraph as pg
import sys

SOURCE_DIR = os.path.dirname(__file__)

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(400,200,500,500)
        self.statusBar()
        openFile = QAction("Open", self)
        # ショートカット設定
        openFile.setShortcut("Ctrl+O")
        # ステータスバー設定 (下に出てくるやつ)
        openFile.setStatusTip("Open new File")
        self.setCentralWidget(MyWidget())
        
        openFile.triggered.connect(self.show_file_dialog)

        # メニューバー作成
        menubar = self.menuBar()
        fileMenu = menubar.addMenu("&File")
        fileMenu.addAction(openFile)

    def show_file_dialog(self):
        # 第二引数はダイアログのタイトル、第三引数は表示するパス
        fname = QFileDialog.getOpenFileName(self, "Open file", SOURCE_DIR)

        # fname[0]は選択したファイルのパス（ファイル名を含む）
        if fname[0]:
            # ファイル読み込み
            with open(fname[0], "r") as f:
                data = f.read()
                print(data)

class MyWidget(QWidget):

    def __init__(self):
        super().__init__()
        # self.setGeometry(400,200,500,500)
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
gui = MyWindow()
gui.show()

sys.exit(app.exec_())  