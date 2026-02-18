# main.py
from window import MainWindow
import sys
from PyQt5 import QtWidgets

def main():
    
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
    

if __name__ == "__main__":
    main()
