"""
主程序
"""
import sys

from myui import *

def main():
    
    app = QApplication(sys.argv)
    newUI = MyUI()
    newUI.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()