"""Inicialização do Grid View Analysis."""

import sys

from PyQt5 import QtWidgets

from window import MainWindow


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Grid View Analysis")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
