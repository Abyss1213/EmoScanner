"""
pyqt5界面驱动
"""

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt,QCoreApplication,QThread
from PyQt5.QtGui import QCursor
from qtui import *
from videoCapture import *
from predict_and_show import *
from utils import free
from PyQt5.Qt import QMessageBox
    
class MyUI(QMainWindow, Ui_Form):
    '''
    主界面
    '''
    def __init__(self, parent=None):
        super(MyUI, self).__init__(parent)
        self.setupUi(self)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.label_1_1.video_time = max_frame_num/frame_rate
        self.dialog = DialogUI()
        self.dialog.show()
        self.dialog.setVisible(False)
        self.dialog.setWindowModality(Qt.ApplicationModal)
        self.signal = 0
    
    def video(self):
        self.signal = 0
        self.pushButton.setVisible(False)
        self.pushButton_2.setVisible(False)
        self.pushButton_3.setVisible(True)
        self.pushButton_4.setVisible(True)
        QApplication.processEvents()
        videoCapture(self,0)
        if(self.signal!=-1):
            predict_and_show(self)
            self.pushButton.setVisible(True)
            self.pushButton_2.setVisible(True)
            self.pushButton_3.setVisible(True)
            self.pushButton_4.setVisible(True)
    
    def load(self):
        self.signal=2
        self.label_1_1.setText('拖入窗口')
        self.label_1_1.setAcceptDrops(True)
        self.pushButton.setVisible(False)
        self.pushButton_2.setVisible(False)
        self.pushButton_3.setVisible(True)
        self.pushButton_4.setVisible(True)
        QApplication.processEvents()
            
    def start(self):
        self.pushButton_3.setVisible(False)
        self.pushButton_4.setVisible(False)
        self.label_1_3.setText("获取中......")
        if(self.signal==0):
            self.dialog.setVisible(True)
            self.signal = 1  
        elif(self.signal==2):
            self.signal = 1
            videoCapture(self,self.label_1_1.video_file)
            if(self.signal!=-1):
                predict_and_show(self)
                self.pushButton.setVisible(True)
                self.pushButton_2.setVisible(True)
                self.pushButton_3.setVisible(True)
                self.pushButton_4.setVisible(True)
    
    def undo(self):
        self.signal=-1
        self.label_1_1.setText("")
        self.label_1_1.setPixmap(QtGui.QPixmap("ui_pictures/1.png"))
        self.pushButton.setVisible(True)
        self.pushButton_2.setVisible(True)
        self.pushButton_3.setVisible(False)
        self.pushButton_4.setVisible(False)

    def myhide(self):
        self.showMinimized()
        
    def myclose(self):
        self.signal = -1
        QApplication.quit()
    
    #隐藏自带标题栏需要添加捕捉鼠标实现拖动窗口
    def mousePressEvent(self, event):
        if event.button()==Qt.LeftButton:
            self.m_flag=True
            self.m_Position=event.globalPos()-self.pos() #获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))  #更改鼠标图标
            
    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:  
            self.move(QMouseEvent.globalPos()-self.m_Position)#更改窗口位置
            QMouseEvent.accept()
            
    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag=False
        self.setCursor(QCursor(Qt.ArrowCursor))
        
class DialogUI(QMainWindow, Ui_Dialog):
    '''
    弹出框
    '''
    def __init__(self, parent=None):
        super(DialogUI, self).__init__(parent)
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setupUi(self)
        
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    newUI = MyUI()
    newUI.show()
    sys.exit(app.exec_())
    
