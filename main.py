import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsPixmapItem, QGraphicsItem, QGraphicsScene
from PyQt5.QtCore import QTimer, pyqtSignal, QObject
from mainWindow import Ui_MainWindow
from layer import CenterTrackLyaer, PointCloudLayer, BackGroundLayer, Layer
from detail import detailWindow

class mainWindow(QMainWindow, Ui_MainWindow,QObject):
    send_result_to_detail_window = pyqtSignal(list)
    send_pos_to_detail_window = pyqtSignal(tuple)

    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        # 创建detail窗口
        self.detail = detailWindow()
        self.send_result_to_detail_window.connect(self.detail.receive_result)
        self.send_pos_to_detail_window.connect(self.detail.receive_click_event)

        Layer.output_height = self.image_here.height()
        Layer.output_width = self.image_here.width()

        # 绑定函数和快捷键
        self.CenterTrack.clicked.connect(self.CenterTrackOnClick)
        self.PointCloud.clicked.connect(self.PointCloudOnClick)
        self.nextOne.clicked.connect(self.OnClickNextOne)
        self.lastOne.clicked.connect(self.OnClickLastOne)
        self.playOrPause.clicked.connect(self.onClickplayOrPause)
        self.CenterTrackScore.valueChanged.connect(self.OnCenterTrackScoreChange)
        self.CenterTrackCombox.currentIndexChanged.connect(self.onCenterTrackComboxChange)
        self.PointCloud_show_box.clicked.connect(self.onPointCloud_show_box)
        self.image_here.mousePressEvent = lambda event:\
            self.send_pos_to_detail_window.emit((
                event.pos().x()*Layer.shape[1]//Layer.output_width,
                event.pos().y()*Layer.shape[0]//Layer.output_height))
        # 初始化变量
        self.id = 0
        self.scene = QGraphicsScene()
        self.image_here.setScene(self.scene)


        print(self.image_here.height())
        print(self.image_here.width())
        self.timer = QTimer()
        self.timer.timeout.connect(self.onTimeoutChangeFrame)
        # 切换到第一张图片
        self.switch(self.id)
        self.detail.show()

    def CenterTrackOnClick(self):
        self.update()

    def PointCloudOnClick(self):
        self.update()

    def onClickplayOrPause(self):
        if (self.timer.isActive()):
            self.timer.stop()
        else:
            if (self.id <= 485):
                self.timer.start(100)

    def onTimeoutChangeFrame(self):
        new_id = self.id + 1
        if (new_id <= 485):
            self.id = new_id
        else:
            self.timer.stop()
        self.switch(self.id)

    def switch(self, id: int):
        if (hasattr(self, "img")):
            del self.img
        if (hasattr(self, "CenterTrackLayer")):
            del self.CenterTrackLayer
        if (hasattr(self, "PointCloudLayer")):
            del self.PointCloudLayer

        # 保存的源矩阵
        self.img = cv2.imread(r"E:\CenterFusion\img\img_{}.jpeg".format(id))
        # 输出层
        self.display = BackGroundLayer(self.img)

        self.CenterTrackLayer = CenterTrackLyaer(id, self.CenterTrackScore.value(),
                                                 self.CenterTrackCombox.currentIndex())
        self.PointCloudLayer = PointCloudLayer(id + 1, self.PointCloud_show_box.isChecked())

        # 拼接所有图层
        self.update(clean=False)

        self.send_result_to_detail_window.emit(self.CenterTrackLayer.results)

    def update(self, clean=True):
        self.display = BackGroundLayer(self.img) if clean else self.display
        if self.CenterTrack.isChecked():
            self.display += self.CenterTrackLayer
        if self.PointCloud.isChecked():
            self.display += self.PointCloudLayer
        if not self.scene.items():
            self.scene.addPixmap(self.display.toQPixmap())
        else:
            self.scene.items().pop()
            self.scene.addPixmap(self.display.toQPixmap())

    def OnClickNextOne(self):
        new_id = self.id + 1
        if (new_id <= 485):
            self.id = new_id
        self.switch(self.id)

    def OnClickLastOne(self):
        new_id = self.id - 1
        if (new_id >= 0):
            self.id = new_id
        self.switch(self.id)

    def OnCenterTrackScoreChange(self, val):
        self.CenterTrackLayer.condition["threshold"] = val
        self.update()

    def onCenterTrackComboxChange(self, val):
        self.CenterTrackLayer.condition["type"] = val
        self.update()

    def onPointCloud_show_box(self):
        self.PointCloudLayer.condition["show_box"] = self.PointCloud_show_box.isChecked()
        self.update()



if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 显示窗口
    win = mainWindow()
    win.show()
    sys.exit(app.exec_())
