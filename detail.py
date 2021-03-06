from detailForm import Ui_Form
from PyQt5.QtWidgets import QDialog, QMainWindow
from PyQt5.QtCore import QObject, pyqtSlot
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt
from utils import CENTERFUSION_CLASS_NAME
import math
import numpy as np


class detail_slot_adapter(QObject):
    def __init__(self, result):
        self.result = result


class detailWindow(QMainWindow, Ui_Form, QObject):
    def __init__(self):
        super(detailWindow, self).__init__()
        self.setupUi(self)
        self.model = QStandardItemModel()
        self.tableView.setModel(self.model)
        self.PointsEuclideanDist = lambda x1, y1, x2, y2: np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
        self.EuclideanDist = lambda x, y, z: np.sqrt(x ** 2 + y ** 2 + z ** 2)
        self.header = ["id", "score", "类别", "位置", "距离", "rot_y", "alpha角", "速度向量", "速度", "在图中中心点"]
        self.threshold = 0.5

    def toStr(self, x):
        if (isinstance(x, np.float64) or isinstance(x, np.float32) or isinstance(x, float)):
            return str(round(x, 4))
        elif isinstance(x, np.ndarray) or isinstance(x, list):
            return str(np.round(x, 2))
        else:
            return str(x)

    def receive_result(self, data: list):
        self.results = data
        self.update_table()

    def receive_threshold(self, data: float):
        self.threshold = data

    def receive_click_event(self, pos):
        p = self.closest_point(pos, self.threshold)
        if p != -1:
            self.tableView.selectRow(p)

    def onHeaderClick(self, x: int):
        if self.onHeaderClickRecord[x] == 0 or self.onHeaderClickRecord[x] == -1:
            self.model.sort(x, Qt.AscendingOrder)
            self.onHeaderClickRecord[x] = 1
        else:
            self.model.sort(x, Qt.DescendingOrder)
            self.onHeaderClickRecord[x] = -1

    def update_table(self):
        self.model.clear()
        self.model.setHorizontalHeaderLabels(self.header)
        self.onHeaderClickRecord = [0] * len(self.header)  # 0表示未排序，1表示升序，-1表示降序
        self.tableView.horizontalHeader().sectionClicked.connect(self.onHeaderClick)
        self.tableView.setColumnWidth(1, 50)
        self.tableView.setColumnWidth(3, 150)
        self.tableView.setColumnWidth(7, 150)
        self.tableView.setColumnHidden(0, True)

        for i, result in enumerate(self.results):
            try:
                self.model.appendRow(
                    (
                        QStandardItem(self.toStr(i)) for i in
                        (
                            i,
                            result["score"],
                            CENTERFUSION_CLASS_NAME[result["class"] - 1],
                            result["loc"],
                            self.EuclideanDist(*result["loc"]),
                            math.degrees(result["rot_y"]),
                            math.degrees(result["alpha"]),
                            result["velocity"],
                            self.EuclideanDist(*result["velocity"]),
                            result["ct"]
                        )
                    )
                )
            except Exception as e:
                print(e)

    def closest_point(self, pos, threshold):
        """
        找到最接近的点
        :param pos:
        :return:
        """
        minn = (-1, np.inf)
        for i in range(self.model.rowCount(self.tableView.rootIndex())):
            index = int(self.model.index(i, 0, self.tableView.rootIndex()).data())
            if (self.results[index]["score"] < threshold): continue
            lx, ly, rx, ry = self.results[index]["bbox"]
            x, y = pos
            if (x < lx or y < ly or x > rx or y > ry): continue
            _ = self.PointsEuclideanDist(*self.results[index]["ct"], *pos)
            if (_ < minn[1]):
                minn = i, _
        return minn[0]
