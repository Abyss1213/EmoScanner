"""
qtdesigner自动生成代码
"""

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'generate.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!

import PyQt5.sip
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWebEngineWidgets import QWebEngineView

import cv2

class MyLabel(QtWidgets.QLabel):
    '''
    继承label，实现label可捕获鼠标拖入事件
    '''
    def __init__(self, parent=None):
        super(MyLabel, self).__init__(parent)
        self.video_file = ''
        self.video_time = 0
        
    # 鼠标拖入事件
    def dragEnterEvent(self, evn):
        self.setStyleSheet("background-color: rgb(168, 193, 221);")
        evn.accept()

    # 鼠标放开执行
    def dropEvent(self, evn):
        #self.setText('文件路径：\n' + evn.mimeData().text())
        self.video_file = evn.mimeData().text()[8:]
        video_captor = cv2.VideoCapture(self.video_file)
        frame_rate = video_captor.get(5)   # 帧速率
        FrameNumber = video_captor.get(7)  # 视频文件的帧数
        self.video_time = FrameNumber/frame_rate
        #print(self.video_time)
        ret, frame = video_captor.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg))
        self.setStyleSheet("background-color: rgb(245, 245, 245);")
        
        video_captor.release()
        
class MyWidget(QtWidgets.QWidget):
    '''
    继承widget，实现widget可捕获鼠标拖入事件
    '''
    def __init__(self, parent=None):
        super(MyWidget, self).__init__(parent)
        self.view = QWebEngineView()
        #self.view.load(QtCore.QUrl("file:///second_line.html"))
        #self.view.load(QtCore.QUrl("file:///pie.html"))
        testlo = QtWidgets.QGridLayout()
        
        self.setLayout(testlo)
        testlo.addWidget(self.view)
        #view.setVisible(False)


'''
qtdesigner自动生成代码
'''
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'generate.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.setWindowModality(QtCore.Qt.ApplicationModal)
        Form.resize(972, 637)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
        Form.setSizePolicy(sizePolicy)
        self.tabWidget = QtWidgets.QTabWidget(Form)
        self.tabWidget.setGeometry(QtCore.QRect(-2, 30, 981, 611))
        self.tabWidget.setElideMode(QtCore.Qt.ElideNone)
        self.tabWidget.setTabBarAutoHide(False)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.label_1 = QtWidgets.QLabel(self.tab)
        self.label_1.setGeometry(QtCore.QRect(1, -6, 981, 591))
        self.label_1.setStyleSheet("")
        self.label_1.setText("")
        self.label_1.setScaledContents(True)
        self.label_1.setAlignment(QtCore.Qt.AlignCenter)
        self.label_1.setObjectName("label_1")
        self.label_1_3 = QtWidgets.QLabel(self.tab)
        self.label_1_3.setGeometry(QtCore.QRect(750, 560, 218, 21))
        self.label_1_3.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_1_3.setText("")
        self.label_1_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_1_3.setObjectName("label_1_3")
        self.label_1_1 = MyLabel(self.tab)
        self.label_1_1.setGeometry(QtCore.QRect(40, 40, 640, 480))
        self.label_1_1.setAcceptDrops(False)
        self.label_1_1.setStyleSheet("background-color: rgb(245, 245, 245);")
        self.label_1_1.setText("")
        self.label_1_1.setPixmap(QtGui.QPixmap("ui_pictures/1.png"))
        self.label_1_1.setScaledContents(True)
        self.label_1_1.setAlignment(QtCore.Qt.AlignCenter)
        self.label_1_1.setObjectName("label_1_1")
        self.label_1_2 = QtWidgets.QLabel(self.tab)
        self.label_1_2.setGeometry(QtCore.QRect(741, 214, 161, 161))
        self.label_1_2.setText("")
        self.label_1_2.setScaledContents(True)
        self.label_1_2.setObjectName("label_1_2")
        self.pushButton_2 = QtWidgets.QPushButton(self.tab)
        self.pushButton_2.setGeometry(QtCore.QRect(101, 555, 101, 30))
        self.pushButton_2.setStyleSheet("background-color: rgb(85, 85, 255);\n"
"color: rgb(255, 255, 255);")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.tab)
        self.pushButton_3.setGeometry(QtCore.QRect(0, 555, 101, 30))
        self.pushButton_3.setStyleSheet("background-color: rgb(85, 85, 255);\n"
"color: rgb(255, 255, 255);")
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton = QtWidgets.QPushButton(self.tab)
        self.pushButton.setGeometry(QtCore.QRect(0, 555, 101, 30))
        self.pushButton.setStyleSheet("background-color: rgb(85, 85, 255);\n"
"color: rgb(255, 255, 255);")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_4 = QtWidgets.QPushButton(self.tab)
        self.pushButton_4.setGeometry(QtCore.QRect(101, 555, 101, 30))
        self.pushButton_4.setStyleSheet("background-color: rgb(85, 85, 255);\n"
"color: rgb(255, 255, 255);")
        self.pushButton_4.setObjectName("pushButton_4")
        self.label_1.raise_()
        self.label_1_1.raise_()
        self.label_1_3.raise_()
        self.label_1_2.raise_()
        self.pushButton_3.raise_()
        self.pushButton.raise_()
        self.pushButton_4.raise_()
        self.pushButton_2.raise_()
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.label_2_3 = QtWidgets.QLabel(self.tab_2)
        self.label_2_3.setGeometry(QtCore.QRect(660, 204, 301, 61))
        self.label_2_3.setText("")
        self.label_2_3.setScaledContents(True)
        self.label_2_3.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_2_3.setWordWrap(True)
        self.label_2_3.setObjectName("label_2_3")
        self.label_2 = QtWidgets.QLabel(self.tab_2)
        self.label_2.setGeometry(QtCore.QRect(1, -6, 981, 591))
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap("ui_pictures/2.png"))
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName("label_2")
        self.label_2_4 = QtWidgets.QLabel(self.tab_2)
        self.label_2_4.setGeometry(QtCore.QRect(660, 294, 301, 71))
        self.label_2_4.setText("")
        self.label_2_4.setScaledContents(True)
        self.label_2_4.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_2_4.setWordWrap(True)
        self.label_2_4.setObjectName("label_2_4")
        self.label_2_5 = QtWidgets.QLabel(self.tab_2)
        self.label_2_5.setGeometry(QtCore.QRect(660, 392, 301, 81))
        self.label_2_5.setText("")
        self.label_2_5.setScaledContents(True)
        self.label_2_5.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_2_5.setWordWrap(True)
        self.label_2_5.setObjectName("label_2_5")
        self.label_2_6 = QtWidgets.QLabel(self.tab_2)
        self.label_2_6.setGeometry(QtCore.QRect(660, 510, 301, 61))
        self.label_2_6.setText("")
        self.label_2_6.setScaledContents(True)
        self.label_2_6.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_2_6.setWordWrap(True)
        self.label_2_6.setObjectName("label_2_6")
        self.label_2_2_1 = QtWidgets.QLabel(self.tab_2)
        self.label_2_2_1.setGeometry(QtCore.QRect(740, 52, 81, 21))
        self.label_2_2_1.setText("")
        self.label_2_2_1.setScaledContents(True)
        self.label_2_2_1.setWordWrap(True)
        self.label_2_2_1.setObjectName("label_2_2_1")
        self.label_2_2_2 = QtWidgets.QLabel(self.tab_2)
        self.label_2_2_2.setGeometry(QtCore.QRect(740, 86, 81, 21))
        self.label_2_2_2.setText("")
        self.label_2_2_2.setScaledContents(True)
        self.label_2_2_2.setWordWrap(True)
        self.label_2_2_2.setObjectName("label_2_2_2")
        self.label_2_2_3 = QtWidgets.QLabel(self.tab_2)
        self.label_2_2_3.setGeometry(QtCore.QRect(740, 118, 81, 21))
        self.label_2_2_3.setStyleSheet("color: rgb(255, 117, 117);\n"
"font: 75 9pt \"微软雅黑\";")
        self.label_2_2_3.setText("")
        self.label_2_2_3.setScaledContents(True)
        self.label_2_2_3.setWordWrap(True)
        self.label_2_2_3.setObjectName("label_2_2_3")
        self.widget = MyWidget(self.tab_2)
        self.widget.setGeometry(QtCore.QRect(0, 50, 640, 480))
        self.widget.setObjectName("widget")
        self.label_2.raise_()
        self.widget.raise_()
        self.label_2_3.raise_()
        self.label_2_4.raise_()
        self.label_2_5.raise_()
        self.label_2_6.raise_()
        self.label_2_2_1.raise_()
        self.label_2_2_2.raise_()
        self.label_2_2_3.raise_()
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.label_3 = QtWidgets.QLabel(self.tab_3)
        self.label_3.setGeometry(QtCore.QRect(0, 0, 981, 591))
        self.label_3.setText("")
        self.label_3.setPixmap(QtGui.QPixmap("ui_pictures/3.png"))
        self.label_3.setScaledContents(True)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setWordWrap(True)
        self.label_3.setObjectName("label_3")
        self.label_3_2 = QtWidgets.QLabel(self.tab_3)
        self.label_3_2.setGeometry(QtCore.QRect(650, 50, 81, 31))
        self.label_3_2.setStyleSheet("background-color: rgb(231, 198, 198);\n"
"color: rgb(255, 79, 79);")
        self.label_3_2.setScaledContents(True)
        self.label_3_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3_2.setWordWrap(True)
        self.label_3_2.setObjectName("label_3_2")
        self.label_3_7 = QtWidgets.QLabel(self.tab_3)
        self.label_3_7.setGeometry(QtCore.QRect(650, 520, 281, 51))
        self.label_3_7.setText("")
        self.label_3_7.setScaledContents(True)
        self.label_3_7.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_3_7.setWordWrap(True)
        self.label_3_7.setObjectName("label_3_7")
        self.label_3_3 = QtWidgets.QLabel(self.tab_3)
        self.label_3_3.setGeometry(QtCore.QRect(650, 140, 271, 71))
        self.label_3_3.setText("")
        self.label_3_3.setScaledContents(True)
        self.label_3_3.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_3_3.setWordWrap(True)
        self.label_3_3.setObjectName("label_3_3")
        self.label_3_4 = QtWidgets.QLabel(self.tab_3)
        self.label_3_4.setGeometry(QtCore.QRect(650, 250, 271, 41))
        self.label_3_4.setText("")
        self.label_3_4.setScaledContents(True)
        self.label_3_4.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_3_4.setWordWrap(True)
        self.label_3_4.setObjectName("label_3_4")
        self.label_3_5 = QtWidgets.QLabel(self.tab_3)
        self.label_3_5.setGeometry(QtCore.QRect(650, 320, 271, 51))
        self.label_3_5.setText("")
        self.label_3_5.setScaledContents(True)
        self.label_3_5.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_3_5.setWordWrap(True)
        self.label_3_5.setObjectName("label_3_5")
        self.label_3_6 = QtWidgets.QLabel(self.tab_3)
        self.label_3_6.setGeometry(QtCore.QRect(650, 400, 271, 81))
        self.label_3_6.setText("")
        self.label_3_6.setScaledContents(True)
        self.label_3_6.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_3_6.setWordWrap(True)
        self.label_3_6.setObjectName("label_3_6")
        self.widget_2 = MyWidget(self.tab_3)
        self.widget_2.setGeometry(QtCore.QRect(0, 50, 640, 480))
        self.widget_2.setObjectName("widget_2")
        self.tabWidget.addTab(self.tab_3, "")
        self.title = QtWidgets.QLabel(Form)
        self.title.setGeometry(QtCore.QRect(0, 0, 981, 32))
        self.title.setStyleSheet("background-color: rgb(68, 138, 255);\n"
"font: 9pt \"微软雅黑\";\n"
"color: rgb(255, 255, 255);")
        self.title.setScaledContents(True)
        self.title.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.title.setObjectName("title")
        self.hide = QtWidgets.QPushButton(Form)
        self.hide.setGeometry(QtCore.QRect(920, 0, 51, 32))
        self.hide.setStyleSheet("background-color: rgb(68, 138, 255);\n"
"color: rgb(255, 255, 255);")
        self.hide.setFlat(True)
        self.hide.setObjectName("hide")
        self.close = QtWidgets.QPushButton(Form)
        self.close.setGeometry(QtCore.QRect(870, 0, 51, 32))
        self.close.setStyleSheet("background-color: rgb(68, 138, 255);\n"
"color: rgb(255, 255, 255);")
        self.close.setFlat(True)
        self.close.setObjectName("close")

        self.retranslateUi(Form)
        self.tabWidget.setCurrentIndex(0)
        self.pushButton_2.clicked.connect(Form.load)
        self.pushButton.clicked.connect(Form.video)
        self.pushButton_3.clicked.connect(Form.start)
        self.close.clicked.connect(Form.myhide)
        self.hide.clicked.connect(Form.myclose)
        self.pushButton_4.clicked.connect(Form.undo)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "基于微表情的情绪实时分析"))
        self.pushButton_2.setText(_translate("Form", "加载"))
        self.pushButton_3.setText(_translate("Form", "开始"))
        self.pushButton.setText(_translate("Form", "录制"))
        self.pushButton_4.setText(_translate("Form", "返回"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Form", "测试画面"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Form", "情绪分析"))
        self.label_3_2.setText(_translate("Form", "惊讶"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("Form", "波动分析"))
        self.title.setText(_translate("Form", " 基于微表情的情绪实时分析"))
        self.hide.setText(_translate("Form", "×"))
        self.close.setText(_translate("Form", "-"))

class Ui_Dialog(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(525, 238)
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(0, 0, 524, 239))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("ui_pictures/4.png"))
        self.label.setScaledContents(True)
        self.label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label.setWordWrap(True)
        self.label.setObjectName("label")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "注意"))

