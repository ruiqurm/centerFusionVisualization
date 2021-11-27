import cv2
import numpy as np
import pickle
import pycocotools.coco as coco
from nuscenes.utils.geometry_utils import view_points
from PyQt5.QtGui import QImage, QPixmap
from utils import compute_box_3d,project_to_image,draw_box_3d,CENTERFUSION_CLASS_NAME
mycoco = coco.COCO(r"E:\CenterFusion\results\mini_val.json")
def layer_add(background, foregound, offset=1):
    """
    场景叠加
    """
    foregound = foregound.astype(np.uint8)
    img2gray = cv2.cvtColor(foregound, cv2.COLOR_BGR2GRAY)  # 将图片灰度化
    ret, mask = cv2.threshold(img2gray, offset, 255, cv2.THRESH_BINARY)  # ret是阈值（175）mask是二值化图像
    mask_inv = cv2.bitwise_not(mask)  # 获取把logo的区域取反 按位运算
    img1_bg = cv2.bitwise_and(background, background, mask=mask_inv)

    # 取 roi 中与 mask_inv 中不为零的值对应的像素的值，其他值为 0 。
    # 把logo放到图片当中
    img2_fg = cv2.bitwise_and(foregound, foregound, mask=mask)  # 获取logo的像素信息
    img2_fg = img2_fg.astype(np.uint8)
    return cv2.add(img1_bg, img2_fg)  # 相加即可


class Layer:
    output_width = 1600
    output_height = 900
    shape = (900, 1600, 3)

    def __init__(self):
        self._mask = np.zeros(self.shape, np.uint8)
        self._lasttime_condition = dict()
        self.condition = dict()

    def get_mask(self) -> np.ndarray:
        if self._lasttime_condition != self.condition:
            self._create_mask()
        return self._mask

    def _create_mask(self):
        raise NotImplementedError

    # def __add__(self, other: "Layer") -> "Layer":
    #     return layer_add(self.get_mask(), other.get_mask())

    def __iadd__(self, other: "Layer") -> "Layer":
        mask = self.get_mask()
        self._mask = layer_add(mask, other.get_mask())
        return self

    def toQPixmap(self) -> QPixmap:
        """
        # 将图片转换为QPixmap方便显示
        :param img: NDArray风格矩阵
        :return:QPixmap
        """
        img = self._mask
        temp_imgSrc = QImage(img[:], img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
        return QPixmap.fromImage(temp_imgSrc).scaled(self.output_width, self.output_height)


class BackGroundLayer(Layer):
    def __init__(self, img: np.ndarray):
        self._mask = img.copy()

    def get_mask(self) -> np.ndarray:
        return self._mask

    def _create_mask(self):
        pass


class CenterTrackLyaer(Layer):


    def __init__(self, id, threshold=0.4, type=0):
        super(CenterTrackLyaer, self).__init__()
        self.id = id
        filename = r"E:\CenterFusion\results\result_{}.pkl".format(id)
        with open(filename, "rb")as f:
            self.results = pickle.load(f)
        self.condition = {
            "threshold": threshold,
            "type": type
        }

    def _add_coco_bbox(self, img, bbox, cat, conf=1, show_txt=True,
                       no_bbox=False, img_id='default', dist=-1):
        bbox = np.array(bbox, dtype=np.int32)
        dist = ', {:.1f}m'.format(int(dist)) if dist >= 0 else ''
        cat = int(cat)
        c = np.array([0, 127, 127])
        c = (255 - np.array(c)).tolist()
        txt = '{}{:.1f}{}'.format(CENTERFUSION_CLASS_NAME[cat], conf, dist)
        thickness = 2
        fontsize = 0.8
        font = cv2.FONT_HERSHEY_SIMPLEX
        cat_size = cv2.getTextSize(txt, font, fontsize, thickness)[0]
        if not no_bbox:
            cv2.rectangle(
                img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                c, thickness)
        if show_txt:
            cv2.rectangle(img,
                          (bbox[0], bbox[1] - cat_size[1] - thickness),
                          (bbox[0] + cat_size[0], bbox[1]), c, -1)
            cv2.putText(img, txt, (bbox[0], bbox[1] - thickness - 1),
                        font, fontsize, (67, 52, 235), thickness=2, lineType=cv2.LINE_AA)
        return img

    def _create_mask(self):
        self._mask = np.zeros(self.shape)
        if (self.condition["type"] == 0):
            for i in self.results:
                if i["score"] < self.condition["threshold"]: continue
                image = self._add_coco_bbox(self._mask, i['bbox'], i['class'] - 1, i['score'], img_id='generic')
        elif (self.condition["type"] == 1):
            for i in self.results:
                if i["score"] < self.condition["threshold"]: continue
                box_3d = compute_box_3d(i["dim"], i["loc"], i["rot_y"])
                box_2d = project_to_image(box_3d,mycoco.loadImgs(self.id+1)[0]['calib'])
                self._mask = draw_box_3d(self._mask, box_2d.astype(np.int32))
        self._lasttime_condition["threshold"] = self.condition["threshold"]





class PointCloudLayer(Layer):
    def __init__(self, id,show_box = False):
        super(PointCloudLayer, self).__init__()
        self.img_info = mycoco.loadImgs(id)[0]  # may lead to KerError
        self.condition["show_box"] = show_box
        input_w = self.condition["input_w"] = 800
        input_h = self.condition["input_h"] = 448
        self.radar_pc = np.array(self.img_info.get('radar_pc', None))
        pc_2d, mask = self._map_pointcloud_to_image(self.radar_pc, np.array(self.img_info['camera_intrinsic']),
                                                    img_shape=(self.img_info['width'], self.img_info['height']))
        pc_3d = self.radar_pc[:, mask]
        ind = np.argsort(pc_2d[2, :])
        self.pc_2d = pc_2d[:, ind]
        self.pc_3d = pc_3d[:, ind]



        # inp_trans = np.array([[0.5, -0., 0.], [0., 0.5, -1.]])
        # pc_inp, _ = self._transform_pc(pc_2d, inp_trans, input_w, input_h)
        # pc_inp[0:2, :] *= 2
        # for i, p in enumerate(pc_inp[:3, :].T):
        #     color = int((p[2].tolist() / 60.0) * 255)
        #     color = [32, 192, color]
        #     self._mask = cv2.circle(self._mask, (int(p[0]), int(p[1])), 10, color, -1)

    def _create_mask(self):
        self._mask = np.zeros(self.shape)
        output_w = 200
        output_h = 112
        pillar_dim = [1.5, 0.2, 0.2]  # 框的大小，第一个是z轴方向大小
        ry = 0
        for i,v in enumerate(self.pc_3d[:3, :].T):
            if self.condition["show_box"]:
                box_3d = compute_box_3d(dim=pillar_dim, location=v, rotation_y=ry)  # 返回8个点，分别对应box的3D位置
                box_2d = project_to_image(box_3d, self.img_info['calib']).T  # 把3D的盒子映射到2D平面上,返回8个二维的点，第二个信息是图片的信息
                pillar_wh = np.zeros((2, self.pc_3d.shape[1]))
                box_2d_t, m = self._transform_pc(box_2d, np.array([[0.125, -0., 0.],
                                                                   [0., 0.125, -0.25]]), output_w, output_h)
                bbox = [np.min(box_2d_t[0, :]),
                        np.min(box_2d_t[1, :]),
                        np.max(box_2d_t[0, :]),
                        np.max(box_2d_t[1, :])]  # format: xyxy
                pillar_wh[0, i] = bbox[2] - bbox[0]
                pillar_wh[1, i] = bbox[3] - bbox[1]
                pill_wh_inp = pillar_wh * (self.condition["input_w"] / output_w)
                pill_wh_ori = pill_wh_inp * 2
                rect_tl_ori = (
                    np.min(int(self.pc_2d[0, i] - pill_wh_ori[0, i] / 2), 0), np.min(int(self.pc_2d[1, i] - pill_wh_ori[1, i]), 0))
                rect_br_ori = (np.min(int(self.pc_2d[0, i] + pill_wh_ori[0, i] / 2), 0), int(self.pc_2d[1, i]))
                cv2.rectangle(self._mask, rect_tl_ori, rect_br_ori, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            color = 255 - int((v[2].tolist() / 60.0) * 255)
            self._mask = cv2.circle(self._mask, (int(self.pc_2d[0, i]), int(self.pc_2d[1, i])), 6, (255, color, 0), -1)

    def _map_pointcloud_to_image(self, pc, cam_intrinsic, img_shape=(1600, 900)):
        points = pc

        (width, height) = img_shape
        depths = points[2, :]

        ## Take the actual picture
        points = view_points(points[:3, :], cam_intrinsic, normalize=True)

        ## Remove points that are either outside or behind the camera.
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > 0)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < width - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < height - 1)
        points = points[:, mask]
        points[2, :] = depths[mask]

        return points, mask

    def _transform_pc(self, pc_2d, trans, img_width, img_height, filter_out=True):

        if pc_2d.shape[1] == 0:
            return pc_2d, []

        pc_t = np.expand_dims(pc_2d[:2, :].T, 0)  # [3,N] -> [1,N,2]
        t_points = cv2.transform(pc_t, trans)
        t_points = np.squeeze(t_points, 0).T  # [1,N,2] -> [2,N]

        # remove points outside image
        if filter_out:
            mask = (t_points[0, :] < img_width) \
                   & (t_points[1, :] < img_height) \
                   & (0 < t_points[0, :]) \
                   & (0 < t_points[1, :])
            out = np.concatenate((t_points[:, mask], pc_2d[2:, mask]), axis=0)
        else:
            mask = None
            out = np.concatenate((t_points, pc_2d[2:, :]), axis=0)

        return out, mask
