# main.py
import os
import sys
from PyQt5 import QtWidgets


def main():
    # Permite iniciar o app sem carregar automaticamente um grid base.
    os.environ.setdefault("SFM_START_EMPTY", "1")

    from window import MainWindow

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(start_empty=True)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
