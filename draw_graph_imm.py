#!/usr/bin/env python3
import os
import random
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QFileDialog, QAction, QHBoxLayout, QPushButton, QGridLayout
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QIcon
import numpy as np
import pyqtgraph as pg
import sys
import pandas as pd

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

        self.grid_layout = QGridLayout()

        self.widget1 = MyWidget()
        self.grid_layout.addWidget(self.widget1)
        # self.setCentralWidget(self.widget1)
        
        openFile.triggered.connect(self.show_file_dialog)

        # メニューバー作成
        menubar = self.menuBar()
        fileMenu = menubar.addMenu("&File")
        fileMenu.addAction(openFile)

        # self.setLayout(self.grid_layout)

    def show_file_dialog(self):
        # 第二引数はダイアログのタイトル、第三引数は表示するパス
        fname = QFileDialog.getOpenFileName(self, "Open file", SOURCE_DIR, "*.csv")

        # fname[0]は選択したファイルのパス（ファイル名を含む）
        if fname[0]:
            self.widget1.set_csv_path(fname[0])
            self.widget1.update_data()

class MyWidget(QWidget):

    def __init__(self):
        super().__init__()
        # self.setGeometry(400,200,500,500)
        pg.setConfigOptions(antialias=True)
        pg.setConfigOptions(foreground='k')
        pg.setConfigOptions(background='w')

        win = pg.GraphicsLayoutWidget(self,size=(400,300),border=True)
        win.move(10,50)

        graph = win.addPlot(title="Data")
        graph.setLabel('left',"Score", units="")
        graph.setLabel('bottom',"Episode", units="")

        self.curve = graph.plot(pen=pg.mkPen((120,23,200),width=2))

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.timer.setSingleShot(True)

    
    def update_data(self):
        self.df = pd.read_csv(self.csv_path)
        self.curve.setData(self.df.index, self.df["animals"])
        self.timer.start(1000)
    
    def set_csv_path(self, csv_path: str):
        self.csv_path = csv_path


maxX = 100
app = QApplication(sys.argv)
gui = MyWindow()
gui.show()

sys.exit(app.exec_())  