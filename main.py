import numpy as np
import cv2
import matplotlib.pyplot as plt
import struct


class Hgt_data():
    def __init__(self, img_path: str, lon: int, lat: int, min_ele: int, max_ele: int, SRTM_version: str = "SRTM1"):
        self.img_path: str = img_path
        self.image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        self.img_wide = self.image.shape[1]
        self.img_height = self.image.shape[0]
        self.lon: int = lon  # 经度，取西侧的值
        self.lat: int = lat  # 纬度，取南侧的值
        self.NS: str = "N" if lat >= 0 else "S"
        self.EW: str = "E" if lon >= 0 else "W"
        self.min_ele: int = min_ele
        self.max_ele: int = max_ele
        self.SRTM: str = SRTM_version
        match SRTM_version:
            case "SRTM3":
                self.hgt_size = 1201
            case "SRTM1":
                self.hgt_size = 3601
            case _:
                self.hgt_size = 1201
        self.hgt_array = np.zeros((self.hgt_size, self.hgt_size))
        for x in range(self.hgt_size):
            for y in range(self.hgt_size):
                x_in_img = int(x / self.hgt_size * self.img_wide) if int(x / self.hgt_size * self.img_wide) < self.img_wide else self.img_wide - 1
                y_in_img = int(y / self.hgt_size * self.img_height) if int(y / self.hgt_size * self.img_height) < self.img_wide else self.img_height - 1
                ele = int(self.image[y_in_img, x_in_img] / 255 * (self.max_ele - self.min_ele) + self.min_ele)
                self.hgt_array[y, x] = ele
        # print(self.hgt_array)

    def __getitem__(self, index: [int, int]):
        print(index)
        y, x = index
        return self.hgt_array[y, x]

    def __iter__(self):
        for y in self.hgt_array:
            for x in y:
                yield x

    def get_hgt_array(self):
        return self.hgt_array

    def write_hgt(self, filename):
        fw = open(filename, "wb")
        for ele in self:
            dibawei = struct.pack("B", int(ele) & 0b11111111)
            gaobawei = struct.pack("b", int(ele) >> 8 & 0b01111111)
            # print(gaobawei, dibawei)
            fw.write(gaobawei)
            fw.write(dibawei)
        fw.close()

    def plot(self, ax, lim_min: int = -10, lim_max: int = 5000, lim_step: int = 100):
        # TODO: 西半球和南半球的负数
        lat_range = np.arange(self.lat, self.lat + 1 + 1 / (self.hgt_size - 1), 1 / (self.hgt_size - 1))[::-1]
        lon_range = np.arange(self.lon, self.lon + 1 + 1 / (self.hgt_size - 1), 1 / (self.hgt_size - 1))
        lim = np.arange(lim_min, lim_max, lim_step)  # 最小最大步数
        ax.contourf(lon_range, lat_range, self.hgt_array, lim, cmap='binary')
        return ax


if __name__ == '__main__':
    hgt = Hgt_data("./test.png", 0, 0, -10, 1419, "SRTM1")
    hgt.write_hgt("./test.hgt")
    fig, ax = plt.subplots()
    ax = hgt.plot(ax, -10, 1600, 10)
    plt.show()
