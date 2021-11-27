import numpy as np
import cv2
CENTERFUSION_CLASS_NAME =  [
        'car', 'truck', 'bus', 'trailer',
        'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle',
        'traffic_cone', 'barrier']

def comput_corners_3d(dim, rotation_y):
    # dim: 3
    # location: 3
    # rotation_y: 1
    # return: 8 x 3
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    l, w, h = dim[2], dim[1], dim[0]
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners).transpose(1, 0)
    return corners_3d
def compute_box_3d(dim, location, rotation_y):
      # dim: 3
      # location: 3
      # rotation_y: 1
      # return: 8 x 3
    corners_3d = comput_corners_3d(dim, rotation_y)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(1, 3)
    return corners_3d
def project_to_image(pts_3d, P):
  # pts_3d: n x 3
  # P: 3 x 4
  # return: n x 2
    pts_3d_homo = np.concatenate(
    [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
    pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
    # import pdb; pdb.set_trace()
    return pts_2d


def draw_box_3d(image, corners, c=(255, 0, 255), same_color=False):
    face_idx = [[0,1,5,4],
              [1,2,6, 5],
              [3,0,4,7],
              [2,3,7,6]]
    right_corners = [1, 2, 6, 5] if not same_color else []
    left_corners = [0, 3, 7, 4] if not same_color else []
    thickness = 4 if same_color else 2
    corners = corners.astype(np.int32)
    for ind_f in range(3, -1, -1):
        f = face_idx[ind_f]
        for j in range(4):
          # print('corners', corners)
            cc = c
            if (f[j] in left_corners) and (f[(j+1)%4] in left_corners):
                cc = (255, 0, 0)
            if (f[j] in right_corners) and (f[(j+1)%4] in right_corners):
                cc = (0, 0, 255)
            try:
                cv2.line(image, (corners[f[j], 0], corners[f[j], 1]),
                    (corners[f[(j+1)%4], 0], corners[f[(j+1)%4], 1]), cc, thickness, lineType=cv2.LINE_AA)
            except:
                pass
        if ind_f == 0:
            try:
                cv2.line(image, (corners[f[0], 0], corners[f[0], 1]),
                         (corners[f[2], 0], corners[f[2], 1]), c, 1, lineType=cv2.LINE_AA)
                cv2.line(image, (corners[f[1], 0], corners[f[1], 1]),
                         (corners[f[3], 0], corners[f[3], 1]), c, 1, lineType=cv2.LINE_AA)
            except:
                pass

    return image