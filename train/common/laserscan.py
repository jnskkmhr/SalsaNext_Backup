#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import time

import numpy as np
import math
import random
from scipy.spatial.transform import Rotation as R
import cv2

class LaserScan:
    """Class that contains LaserScan with x,y,z,r"""
    EXTENSIONS_SCAN = ['.bin']

    def __init__(self, project=False, H=64, W=2048, fov_up=15.0, fov_down=-25.0, DA=False,flip_sign=False,rot=False,drop_points=False):
        self.project = project
        self.proj_H = H
        self.proj_W = W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down
        self.DA = False
        self.flip_sign = False
        self.rot = False
        self.drop_points = False
        self.nonlinear = False
        self.nnn = False       #not use
        self.apply = 32
        self.h = 5             #filter height
        self.w = 2             #filter width, but apply only label
        self.closing = False
        self.img_aug = False
        self.reset()
        self.start = time.perf_counter()
        self.unfold = True     #some bugs
        self.vls_noise = False  #provide noise with vls data
        if self.vls_noise:
            self.vls = []
            for i in range(25):
                tmp = "/home/masatosaeki/dataset/dropout/{0:06d}.npy".format(i)
                self.vls.append(np.load(tmp))
        #print("call")

        self.relative = False    #absolute2relative only z

    def conversion(self, x):
        if self.nonlinear:
            x = np.clip(x, 0.0, 1.0)
            pi = (1.0 - (x - 1.0)**2.0) ** (0.5) - 0.1
            ki = x ** 0.5
            return pi
        else:
            return x
    
    def completion_scan_fast(self, apply, h):
        #print("input com")
        #range completion
        mean = np.full((int(apply/h), 2048), -1, dtype=np.float32)
        tmp = self.proj_range[:apply,]
        tmp0 = self.proj_range[apply:,]
        tmp = np.ma.masked_equal(tmp, -1)
        tmp = np.split(tmp, apply/h, axis=0)
        for c,i in enumerate(tmp):
            mean[c] = np.mean(i, axis=0, dtype=np.float32)
        #if np.ma.is_masked(mean):
            #mean = mean.filled(fill_value=-1)
        mean[mean == 0] = -1
        mean = mean.repeat(h, axis=0)
        com = np.concatenate([mean, tmp0], axis=0)
        self.proj_range[self.proj_range == -1] = com[self.proj_range == -1]
        #xyz completion
        mean = np.full((int(apply/h), 2048, 3), -1, dtype=np.float32)
        tmp = self.proj_xyz[:apply,]
        tmp0 = self.proj_xyz[apply:,]
        tmp = np.ma.masked_equal(tmp, -1)
        tmp = np.split(tmp, apply/h, axis=0)
        for c,i in enumerate(tmp):
            mean[c] = np.mean(i, axis=0, dtype=np.float32)
        #if np.ma.is_masked(mean):
            #mean = mean.filled(fill_value=-1)
        mean[mean == 0] = -1
        mean = mean.repeat(h, axis=0)
        com = np.concatenate([mean, tmp0], axis=0)
        self.proj_xyz[self.proj_xyz == -1] = com[self.proj_xyz == -1]
        #remission completion
        mean = np.full((int(apply/h), 2048), -1, dtype=np.float32)
        tmp = self.proj_remission[:apply,]
        tmp0 = self.proj_remission[apply:,]
        tmp = np.ma.masked_equal(tmp, -1)
        tmp = np.split(tmp, apply/h, axis=0)
        for c,i in enumerate(tmp):
            mean[c] = np.mean(i, axis=0, dtype=np.float32)
        #if np.ma.is_masked(mean):
            #mean = mean.filled(fill_value=-1)
        mean[mean == 0] = -1
        mean = mean.repeat(h, axis=0)
        com = np.concatenate([mean, tmp0], axis=0)
        self.proj_remission[self.proj_remission == -1] = com[self.proj_remission == -1]

    def reset(self):
        """ Reset scan members. """
        self.points = np.zeros((0, 3), dtype=np.float32)  # [m, 3]: x, y, z
        self.remissions = np.zeros((0, 1), dtype=np.float32)  # [m ,1]: remission

        # projected range image - [H,W] range (-1 is no data)
        self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)

        # unprojected range (list of depths for each point)
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                                dtype=np.float32)

        # projected remission - [H,W] intensity (-1 is no data)
        self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                      dtype=np.float32)

        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                                dtype=np.int32)

        # for each point, where it is in the range image
        self.proj_x = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: y

        # mask containing for each pixel, if it contains a point or not
        self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                                  dtype=np.int32)  # [H,W] mask

    def size(self):
        """ Return the size of the point cloud. """
        return self.points.shape[0]

    def __len__(self):
        return self.size()

    def open_scan(self, filename):
        """ Open raw scan and fill in attributes
        """
        # reset just in case there was an open structure
        self.reset()

        # check filename is string
        if not isinstance(filename, str):
            raise TypeError("Filename should be string type, "
                            "but was {type}".format(type=str(type(filename))))

        # check extension is a laserscan
        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
            raise RuntimeError("Filename extension is not valid scan file.")

        # if all goes well, open pointcloud
        scan = np.fromfile(filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))

        # put in attribute
        points = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission
        if self.drop_points is not False:
            self.points_to_drop = np.random.randint(0, len(points)-1,int(len(points)*self.drop_points))
            points = np.delete(points,self.points_to_drop,axis=0)
            remissions = np.delete(remissions,self.points_to_drop)
        
        if self.unfold:
            # self.reset()
            projection = self.projection(points, remissions)
            self.proj_range = projection["depth"]
            self.proj_xyz = projection["points"]
            self.channels = projection["channels"]
            self.proj_remission = self.channels[0]
            self.proj_mask[self.proj_range != 1] = 1
            self.proj_mask = self.proj_mask.astype(np.int32)

        else:
            self.set_points(points, remissions)

    def set_points(self, points, remissions=None):
        """ Set scan attributes (instead of opening from file)
        """
        # reset just in case there was an open structure
        self.reset()

        # check scan makes sense
        if not isinstance(points, np.ndarray):
            raise TypeError("Scan should be numpy array")

        # check remission makes sense
        if remissions is not None and not isinstance(remissions, np.ndarray):
            raise TypeError("Remissions should be numpy array")

        # put in attribute
        self.points = points  # get
        if self.flip_sign:
            self.points[:, 1] = -self.points[:, 1]
        if self.DA:
            jitter_x = random.uniform(-5,5)
            jitter_y = random.uniform(-3, 3)
            if not self.relative:
                jitter_z = random.uniform(-1, 0)
            else:
                jitter_z = random.uniform(0, 1)
            self.points[:, 0] += jitter_x
            self.points[:, 1] += jitter_y
            self.points[:, 2] += jitter_z
        if self.rot:
            self.points = self.points @ R.random(random_state=1234).as_dcm().T
        if remissions is not None:
            self.remissions = remissions  # get remission
            #if self.DA:
            #    self.remissions = self.remissions[::-1].copy()
        else:
            self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

        # if projection is wanted, then do it and fill in the structure
        if self.project:
            self.do_range_projection()

    def do_range_projection(self):
        """ Project a pointcloud into a spherical projection image.projection.
            Function takes no arguments because it can be also called externally
            if the value of the constructor was not set (in case you change your
            mind about wanting the projection)
        """
        # laser parameters
        fov_up = self.proj_fov_up / 180.0 * np.pi  # field of view up in rad
        fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # get depth of all points
        depth = np.linalg.norm(self.points, 2, axis=1)

        # get scan components
        scan_x = self.points[:, 0]
        scan_y = self.points[:, 1]
        scan_z = self.points[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

        #conversion
        proj_y = self.conversion(proj_y)

        # scale to image size using angular resolution
        proj_x *= self.proj_W  # in [0.0, W]
        proj_y *= self.proj_H  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
        self.proj_x = np.copy(proj_x)  # store a copy in orig order

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
        self.proj_y = np.copy(proj_y)  # stope a copy in original order

        # copy of depth in original order
        self.unproj_range = np.copy(depth)

        # order in decreasing depth
        self.indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        self.indices = self.indices[order]
        points = self.points[order]
        remission = self.remissions[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # assing to images
        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = points
        self.proj_remission[proj_y, proj_x] = remission
        self.proj_idx[proj_y, proj_x] = self.indices
        self.proj_mask = (self.proj_idx > 0).astype(np.int32)

        if self.relative:
            min_z = np.min(scan_z)
            self.proj_xyz[proj_y, proj_x, 2] = points[:, 2] - min_z
            #print(min_z)

        if self.closing:
            kernel = np.full((self.h, self.w), -1, np.float32)
            self.proj_range = cv2.morphologyEx(self.proj_range, cv2.MORPH_CLOSE, kernel)
            for i in range(3):
                self.proj_xyz[:,:,i] = cv2.morphologyEx(self.proj_xyz[:,:,i], cv2.MORPH_CLOSE, kernel)
            self.proj_remission = cv2.morphologyEx(self.proj_remission, cv2.MORPH_CLOSE, kernel)
            kernel = np.full((self.h, self.w), -1, np.uint8)
            self.proj_mask = cv2.morphologyEx(self.proj_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            self.proj_mask = self.proj_mask.astype(np.int32)

        if self.nnn:
            self.completion_scan_fast(apply=self.apply, h=self.h)

        if self.vls_noise:
            mask = np.full((64, 0) ,0 ,dtype=np.int32)
            while mask.shape[1] < 2048:
                k = np.random.randint(24)
                noise = self.vls[k]
                k = np.random.randint(noise.shape[1])
                noise = noise[:,:k]
                mask = np.concatenate([mask, noise], axis=1)
            mask, _ = np.split(mask, [2048], axis=1)
            self.proj_mask = mask
            self.proj_range = self.proj_range*self.proj_mask
            self.proj_range[self.proj_range == 0] = -1
                
    #yaw
    def get_kitti_columns(self, points: np.array, number_of_columns: int = 2048) -> np.array:
        """ Returns the column indices for unfolding one or more raw KITTI scans """
        azi = np.arctan2(points[..., 1], points[..., 0])
        cols = number_of_columns * (np.pi - azi) / (2 * np.pi)
        # In rare cases it happens that the index is exactly one too large.
        cols = np.minimum(cols, number_of_columns - 1)
        return np.int32(cols)

    #pitch
    def get_kitti_rows(self, points: np.array, threshold: float = 0.0052) -> np.array:
        """ Returns the row indices for unfolding one or more raw KITTI scans """
        azimuth_flipped = np.arctan2(points[..., 1], points[..., 0])  #pitchの算出
        print(azimuth_flipped.shape)
        azi_diffs = np.abs(azimuth_flipped[..., 1:] - azimuth_flipped[..., :-1])  #縦に隣り合っている点の横の相対値の算出
        jump_mask = np.greater(azi_diffs, threshold)  #azi_diffsのほうが大きければTrue, 小さければFalse
        print(jump_mask[jump_mask == True].shape)
        ind = np.add(np.where(jump_mask), 1)  #whereでTrueのindexを返し, それに+1をする
        rows = np.zeros_like(points[..., 0])  #(num of points,)のrowを生成
        rows[..., ind] += 1  #rows[ind]に1をする. (角度がついている所に印をつける.)
        rows = np.int32(np.cumsum(rows, axis=-1))
        #rows -= 64
        rows = np.minimum(rows, self.proj_H - 1)
        if self.nonlinear:
            rows = rows/63
            rows = self.conversion(rows)
            rows = (rows*63).astype(np.int32)
        return rows

    def projection(self, points: np.array, *channels, image_size: tuple = (64, 2048)):
  
        output = {}

        assert points.shape[1] == 3, "Points must contain N xyz coordinates."
        if len(channels) > 0:
            assert all(
                isinstance(x, np.ndarray) for x in channels
            ), "All channels must be numpy arrays."
            assert all(
                x.shape[0] == points.shape[0] for x in channels
            ), "All channels must have the same first dimension as `points`."

        # Get depth of all points for ordering.
        depth_list = np.linalg.norm(points, 2, axis=1)
        self.unproj_range = np.copy(depth_list)
        self.points = np.copy(points)

        # Get the indices of the rows and columns to project to.
        proj_column = self.get_kitti_columns(points, number_of_columns=image_size[1])
        proj_row = self.get_kitti_rows(points)

        print("rows shape: {}, min: {}, max: {}".format(proj_row.shape, np.min(proj_row), np.max(proj_row)))
        print("cols shape: {}, min: {}, max: {}".format(proj_column.shape, np.min(proj_column), np.max(proj_column)))


        if np.any(proj_row >= image_size[0]) or np.any(proj_column >= image_size[1]):
            raise IndexError(
                "Cannot find valid image indices for this point cloud and image size. "
                "Are you sure you entered the correct image size? This function only works "
                "with raw KITTI HDL-64 point clouds (no ego motion corrected data allowed)!"
            )

        # Store a copy in original order.
        output["indices"] = np.stack([np.copy(proj_row), np.copy(proj_column)], axis=-1)

        # Get the indices in order of decreasing depth.
        indices = np.arange(depth_list.shape[0])
        order = np.argsort(depth_list)[::-1]

        indices = indices[order]
        proj_column = proj_column[order]
        proj_row = proj_row[order]

        # Project the points.
        points_img = np.full(shape=(*image_size, 3), fill_value=-1, dtype=np.float32)
        points_img[proj_row, proj_column] = points[order]
        output["points"] = points_img

        # The depth projection.
        depth_img = np.full(shape=image_size, fill_value=-1, dtype=np.float32)
        depth_img[proj_row, proj_column] = depth_list[order]
        output["depth"] = depth_img

        # Convert all channels.
        projected_channels = []
        for channel in channels:
            # Set the shape.
            _shape = (
                (*image_size, channel.shape[1])
                if len(channel.shape) > 1
                else (*image_size,)
            )

            # Initialize the image.
            _image = np.full(shape=_shape, fill_value=-1, dtype=channel.dtype)

            # Assign the values.
            _image[proj_row, proj_column] = channel[order]
            projected_channels.append(_image)
        output["channels"] = projected_channels

        # Get the inverse indices mapping.
        list_indices_img = np.full(image_size, -1, dtype=np.int32)
        list_indices_img[proj_row, proj_column] = indices
        output["inverse"] = list_indices_img

        # Set which points are used in the projection.
        active_list = np.full(depth_list.shape, fill_value=0, dtype=np.int32)
        active_list[list_indices_img] = 1
        output["active"] = active_list.astype(np.bool)

        return output


class SemLaserScan(LaserScan):
    """Class that contains LaserScan with x,y,z,r,sem_label,sem_color_label,inst_label,inst_color_label"""
    EXTENSIONS_LABEL = ['.label']

    def __init__(self, sem_color_dict=None, project=False, H=64, W=2048, fov_up=15.0, fov_down=-25.0, max_classes=300, DA=False,flip_sign=False,drop_points=False):
        super(SemLaserScan, self).__init__(project, H, W, fov_up, fov_down)
        self.reset()

        # make semantic colors
        if sem_color_dict:
            # if I have a dict, make it
            max_sem_key = 0
            for key, data in sem_color_dict.items():
                if key + 1 > max_sem_key:
                    max_sem_key = key + 1
            self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
            for key, value in sem_color_dict.items():
                self.sem_color_lut[key] = np.array(value, np.float32) / 255.0
        else:
            # otherwise make random
            max_sem_key = max_classes
            self.sem_color_lut = np.random.uniform(low=0.0,
                                                   high=1.0,
                                                   size=(max_sem_key, 3))
            # force zero to a gray-ish color
            self.sem_color_lut[0] = np.full((3), 0.1)

        # make instance colors
        max_inst_id = 100000
        self.inst_color_lut = np.random.uniform(low=0.0,
                                                high=1.0,
                                                size=(max_inst_id, 3))
        # force zero to a gray-ish color
        self.inst_color_lut[0] = np.full((3), 0.1)

    def completion_label(self, apply, h, w):
        #print("label com")
        num = self.indices.shape[0]
        for i in range(0,apply,h):
            for j in range(0,2048,w):
                #com proj_sem_label and adjust idx
                tmp = self.proj_sem_label[i:i+h, j:j+w]
                jud = tmp[tmp == 0]
                if (0 < jud.shape[0] < ((h*w))):
                    jud = tmp[tmp != 0]  
                    val, cnt = np.unique(jud, return_counts=True)
                    cnt = np.argmax(cnt)
                    tmp[tmp == 0] = (val[cnt]).astype(np.int32)
                    #adjust idx
                    tmp2 = self.proj_idx[i:i+h, j:j+w]
                    cnt = 0
                    for k in tmp2:
                        if k == -1:
                            tmp2[cnt] = num
                            num += 1
                        cnt += 1
                    #com proj_sem_color
                    for k in range(3):
                        tmp = self.proj_sem_color[i:i+h, j:j+w, k]
                        jud = tmp[tmp == 0]
                        if (0 < jud.shape[0] < ((h*w))):#全埋まりor全なし以外入る
                            jud = tmp[tmp != 0]
                            val, cnt = np.unique(jud, return_counts=True)
                            cnt = np.argmax(cnt)
                            tmp[tmp == 0] = (val[cnt]).astype(np.float)
        #com mask
        self.proj_mask = (self.proj_idx > 0).astype(np.int32)
        #for i in self.proj_mask:  
        #print(i)

    def reset(self):
        """ Reset scan members. """
        super(SemLaserScan, self).reset()

        # semantic labels
        self.sem_label = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: label
        self.sem_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

        # instance labels
        self.inst_label = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: label
        self.inst_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

        # projection color with semantic labels
        self.proj_sem_label = np.zeros((self.proj_H, self.proj_W),
                                       dtype=np.int32)  # [H,W]  label
        self.proj_sem_color = np.zeros((self.proj_H, self.proj_W, 3),
                                       dtype=np.float)  # [H,W,3] color

        # projection color with instance labels
        self.proj_inst_label = np.zeros((self.proj_H, self.proj_W),
                                        dtype=np.int32)  # [H,W]  label
        self.proj_inst_color = np.zeros((self.proj_H, self.proj_W, 3),
                                        dtype=np.float)  # [H,W,3] color

    def open_label(self, filename):
        """ Open raw scan and fill in attributes
        """
        # check filename is string
        if not isinstance(filename, str):
            raise TypeError("Filename should be string type, "
                            "but was {type}".format(type=str(type(filename))))

        # check extension is a laserscan
        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_LABEL):
            raise RuntimeError("Filename extension is not valid label file.")

        # if all goes well, open label
        label = np.fromfile(filename, dtype=np.int32)
        label = label.reshape((-1))

        if self.drop_points is not False:
            label = np.delete(label,self.points_to_drop)
        # set it
        self.set_label(label)

    def set_label(self, label):
        """ Set points for label not from file but from np
        """
        # check label makes sense
        if not isinstance(label, np.ndarray):
            raise TypeError("Label should be numpy array")

        # only fill in attribute if the right size
        if label.shape[0] == self.points.shape[0]:
            self.sem_label = label & 0xFFFF  # semantic label in lower half
            self.inst_label = label >> 16  # instance id in upper half
        else:
            print("Points shape: ", self.points.shape)
            print("Label shape: ", label.shape)
            raise ValueError("Scan and Label don't contain same number of points")

        # sanity check
        assert ((self.sem_label + (self.inst_label << 16) == label).all())

        if self.project:
            self.do_label_projection()

    def colorize(self):
        """ Colorize pointcloud with the color of each semantic label
        """
        self.sem_label_color = self.sem_color_lut[self.sem_label]
        self.sem_label_color = self.sem_label_color.reshape((-1, 3))

        self.inst_label_color = self.inst_color_lut[self.inst_label]
        self.inst_label_color = self.inst_label_color.reshape((-1, 3))

    def do_label_projection(self):
        # only map colors to labels that exist
        mask = self.proj_idx >= 0

        # semantics
        self.proj_sem_label[mask] = self.sem_label[self.proj_idx[mask]]
        self.proj_sem_color[mask] = self.sem_color_lut[self.sem_label[self.proj_idx[mask]]]
        # instances
        self.proj_inst_label[mask] = self.inst_label[self.proj_idx[mask]]
        self.proj_inst_color[mask] = self.inst_color_lut[self.inst_label[self.proj_idx[mask]]]

        if self.closing:
            kernel = np.full((self.h, self.w), -1, np.float32)
            for i in range(3):
                self.proj_sem_color[:,:,i] = cv2.morphologyEx(self.proj_sem_color[:,:,i], cv2.MORPH_CLOSE, kernel)
            kernel = np.full((self.h, self.w), 0, np.uint8)
            self.proj_sem_label = cv2.morphologyEx(self.proj_sem_label.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            self.proj_sem_label = self.proj_sem_label.astype(np.int32)

        if self.nnn:
            self.completion_label(apply=self.apply, h=self.h, w=self.w)

        if self.vls_noise:
            self.proj_sem_label = self.proj_sem_label*self.proj_mask
            for i in range(3):
                self.proj_sem_color[:,:,i] = self.proj_sem_color[:,:,i]*self.proj_mask

        
        if self.img_aug:
            img = ImageAugmentation(ran=self.proj_range,
                                    xyz=self.proj_xyz,
                                    remission=self.proj_remission,
                                    mask=self.proj_mask,
                                    label=self.proj_sem_label)
            if np.random.rand(1) > 1:
                img.dasrc()
            if np.random.rand(1) > 1:
                img.cutoffone()
            if np.random.rand(1) > 1:
                img.cutoffmany()
            if np.random.rand(1) > 1:
                img.gaussianblur()
            if np.random.rand(1) > 1:
                img.downscaleandupscale()
            if np.random.rand(1) > 1:
                img.rot()

            self.proj_range = img.ran
            self.proj_xyz = img.xyz
            self.proj_remission = img.remission
            self.proj_mask = img.mask
            self.proj_sem_label = img.label

        score = time.perf_counter() - self.start
        print(score)
        self.start = time.perf_counter()

class ImageAugmentation():
        
    def __init__(self, ran, xyz, remission, mask, label):
        self.ran = ran
        self.xyz = xyz
        self.remission = remission
        self.mask = mask
        self.label = label

    def randomresizecrop(self):
        k = np.random.randint(2)
        if k == 0:#[32, 1024]
            c1 = 32
            c2 = 1024
            s = 64/c1
            h = np.random.randint(64-c1)
            w = np.random.randint(2048-c2)
            self.ran = self.ran[h:h+c1, w:w+c2].repeat(s, axis=0).repeat(s, axis=1)
            self.xyz = self.xyz[h:h+c1, w:w+c2].repeat(s, axis=0).repeat(s, axis=1)
            self.remission = self.remission[h:h+c1, w:w+c2].repeat(s, axis=0).repeat(s, axis=1)
            self.mask = self.mask[h:h+c1, w:w+c2].repeat(s, axis=0).repeat(s, axis=1)
            self.label = self.label[h:h+c1, w:w+c2].repeat(s, axis=0).repeat(s, axis=1)
        if k == 1:#[16, 512]
            c1 = 16
            c2 = 512
            s = 64/c1
            h = np.random.randint(64-c1)
            w = np.random.randint(2048-c2)
            self.ran = self.ran[h:h+c1, w:w+c2].repeat(s, axis=0).repeat(s, axis=1)
            self.xyz = self.xyz[h:h+c1, w:w+c2].repeat(s, axis=0).repeat(s, axis=1)
            self.remission = self.remission[h:h+c1, w:w+c2].repeat(s, axis=0).repeat(s, axis=1)
            self.mask = self.mask[h:h+c1, w:w+c2].repeat(s, axis=0).repeat(s, axis=1)
            self.label = self.label[h:h+c1, w:w+c2].repeat(s, axis=0).repeat(s, axis=1)

    def rot(self):
        angle = 15
        angle = int(random.uniform(-angle, angle))
        h, w = self.ran.shape
        M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
        self.ran = cv2.warpAffine(self.ran, M, (w, h))
        self.xyz[:,:,0] = cv2.warpAffine(self.xyz[:,:,0], M, (w, h))
        self.xyz[:,:,1] = cv2.warpAffine(self.xyz[:,:,1], M, (w, h))
        self.xyz[:,:,2] = cv2.warpAffine(self.xyz[:,:,2], M, (w, h))
        self.remission = cv2.warpAffine(self.remission, M, (w, h))
        #self.mask = cv2.warpAffine(self.mask, M, (w, h))
        #self.label = cv2.warpAffine(self.label, M, (w, h))

    def dasrc(self):
        print("da-src")
        ran1 = np.full((4, 32, 1024), -1, dtype=np.float32)
        xyz1 = np.full((4, 32, 1024, 3), -1, dtype=np.float32)
        remission1 = np.full((4, 32, 1024), -1, dtype=np.float32)
        mask1 = np.full((4, 32, 1024), -1, dtype=np.int32)
        label1 = np.full((4, 32, 1024), -1, dtype=np.int32)
        for i in range(4):
            k = np.random.randint(3)
            if k == 0:#[32, 1024]
                c1 = 32
                c2 = 1024
                #s = (64/c1)*0.5
                h = np.random.randint(64-c1)
                w = np.random.randint(2048-c2)
                ran1[i] = self.ran[h:h+c1, w:w+c2]
                xyz1[i] = self.xyz[h:h+c1, w:w+c2]
                remission1[i] = self.remission[h:h+c1, w:w+c2]
                mask1[i] = self.mask[h:h+c1, w:w+c2]
                label1[i] = self.label[h:h+c1, w:w+c2]
            if k == 1:#[16, 512]
                c1 = 16
                c2 = 512
                s = (64/c1)*0.5
                h = np.random.randint(64-c1)
                w = np.random.randint(2048-c2)
                ran1[i] = self.ran[h:h+c1, w:w+c2].repeat(s, axis=0).repeat(s, axis=1)
                xyz1[i] = self.xyz[h:h+c1, w:w+c2].repeat(s, axis=0).repeat(s, axis=1)
                remission1[i] = self.remission[h:h+c1, w:w+c2].repeat(s, axis=0).repeat(s, axis=1)
                mask1[i] = self.mask[h:h+c1, w:w+c2].repeat(s, axis=0).repeat(s, axis=1)
                label1[i] = self.label[h:h+c1, w:w+c2].repeat(s, axis=0).repeat(s, axis=1)
            if k == 2:#[64, 2048]
                c1 = 64
                c2 = 2048
                #s = (64/c1)*0.5
                #h = np.random.randint(64-c1)
                #w = np.random.randint(2048-c2)
                ran1[i] = self.ran[::2, ::2]
                xyz1[i] = self.xyz[::2, ::2]
                remission1[i] = self.remission[::2, ::2]
                mask1[i] = self.mask[::2, ::2]
                label1[i] = self.label[::2, ::2]
        
        k0, k1, k2, k3 = np.random.choice(np.arange(4), 4, replace=False)
        ran2 = np.concatenate([ran1[k0], ran1[k1]], axis=0)
        ran3 = np.concatenate([ran1[k2], ran1[k3]], axis=0)
        self.ran = np.concatenate([ran2, ran3], axis=1)
        xyz2 = np.concatenate([xyz1[k0], xyz1[k1]], axis=0)
        xyz3 = np.concatenate([xyz1[k2], xyz1[k3]], axis=0)
        self.xyz = np.concatenate([xyz2, xyz3], axis=1)
        remission2 = np.concatenate([remission1[k0], remission1[k1]], axis=0)
        remission3 = np.concatenate([remission1[k2], remission1[k3]], axis=0)
        self.remission = np.concatenate([remission2, remission3], axis=1)
        mask2 = np.concatenate([mask1[k0], mask1[k1]], axis=0)
        mask3 = np.concatenate([mask1[k2], mask1[k3]], axis=0)
        self.mask = np.concatenate([mask2, mask3], axis=1)
        label2 = np.concatenate([label1[k0], label1[k1]], axis=0)
        label3 = np.concatenate([label1[k2], label1[k3]], axis=0)
        self.label = np.concatenate([label2, label3], axis=1)

    def cutoffone(self):
        print("cut off one")
        c1 = np.random.randint(16, 32)
        c2 = np.random.randint(512, 1024)
        h = np.random.randint(64-c1)
        w = np.random.randint(2048-c2)
        self.ran[h:h+c1, w:w+c2] = -1
        self.xyz[h:h+c1, w:w+c2] = -1
        self.remission[h:h+c1, w:w+c2] = -1
        self.mask[h:h+c1, w:w+c2] = 0
        self.label[h:h+c1, w:w+c2] = 0
        
    def cutoffmany(self):
        print("cut off many")
        k = np.random.randint(32)
        for i in range(k):
            c1 = np.random.randint(4, 16)
            c2 = np.random.randint(128, 512)
            h = np.random.randint(64-c1)
            w = np.random.randint(2048-c2)
            self.ran[h:h+c1, w:w+c2] = -1
            self.xyz[h:h+c1, w:w+c2] = -1
            self.remission[h:h+c1, w:w+c2] = -1
            self.mask[h:h+c1, w:w+c2] = 0
            self.label[h:h+c1, w:w+c2] = 0
        

    def gaussianblur(self):
        print("gaussian blur")
        sigma = 0.3
        k = 5
        self.ran = cv2.GaussianBlur(self.ran, (k,k), sigma)
        self.xyz[:,:,0] = cv2.GaussianBlur(self.xyz[:,:,0], (k,k), sigma)
        self.xyz[:,:,1] = cv2.GaussianBlur(self.xyz[:,:,1], (k,k), sigma)
        self.xyz[:,:,2] = cv2.GaussianBlur(self.xyz[:,:,2], (k,k), sigma)
        self.remission = cv2.GaussianBlur(self.remission, (k,k), sigma)
        #self.mask = cv2.GaussianBlur(self.mask, (k,k), sigma)
        #self.label = cv2.GaussianBlur(self.label, (k,k), sigma)

    def downscaleandupscale(self):
        print("down and up")
        self.ran = self.ran[::2, ::2].repeat(2, axis=0).repeat(2, axis=1)
        self.xyz = self.xyz[::2, ::2].repeat(2, axis=0).repeat(2, axis=1)
        self.remission = self.remission[::2, ::2].repeat(2, axis=0).repeat(2, axis=1)
        self.mask = self.mask[::2, ::2].repeat(2, axis=0).repeat(2, axis=1)
        self.label = self.label[::2, ::2].repeat(2, axis=0).repeat(2, axis=1)

class NoiseLaserScan(LaserScan):
    """Class that contains LaserScan with x,y,z,r,sem_label,sem_color_label,inst_label,inst_color_label"""
    EXTENSIONS_LABEL = ['.bin']

    def __init__(self, project=False, H=-1, W=-1, fov_up=-1, fov_down=-1, max_classes=300, DA=False,flip_sign=False,drop_points=False):
        super(NoiseLaserScan, self).__init__(project, H, W, fov_up, fov_down)
        self.reset()
        self.DA = False
        self.flip_sign = False
        self.rot = False
        self.drop_points = False
        self.nonlinear = True
        self.nnn = False       #not use
        self.apply = 32
        self.h = 5             #filter height
        self.w = 2             #filter width, but apply only label
        self.closing = False
        self.img_aug = False