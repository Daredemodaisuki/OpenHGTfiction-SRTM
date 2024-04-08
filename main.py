import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_hgt_from_img(img_path: str, lon: int, lat: int, min_ele: int, max_ele: int, SRTM_version: str = "SRTM"):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_wide = image.shape[1]
    img_height = image.shape[0]
    match SRTM_version:
        case "SRTM":
            hgt_size = 1201
        case "SRTM1":
            hgt_size = 3601
        case _:
            hgt_size = 1201
    hgt = np.zeros((hgt_size, hgt_size))
    for x in range(hgt_size):
        for y in range(hgt_size):
            x_in_img = int(x / hgt_size * img_wide) if int(x / hgt_size * img_wide) < img_wide else img_wide - 1
            y_in_img = int(y / hgt_size * img_height) if int(y / hgt_size * img_height) < img_wide else img_height - 1
            ele = int(image[y_in_img, x_in_img] / 255 * (max_ele - min_ele) + min_ele)
            hgt[y, x] = ele

    print(hgt)
    return hgt


def plot_d(ax, lon: int, lat: int, ele):
    lat_range = np.arange(lat, lat + 1 / 1200 + 1, 1 / 1200)[::-1]  # 南半球纬度为负数
    lon_range = np.arange(lon, lon + 1 + 1 / 1200, 1 / 1200)
    lim = np.arange(1, 5000, 10) # 最小最大步数
    ax.contourf(lon_range, lat_range, ele, lim, cmap='binary')
    return ax


if __name__ == '__main__':
    hgt = get_hgt_from_img("./test.png", 0, 0, 5, 3455)
    fig, ax = plt.subplots()
    ax = plot_d(ax, 0, 0, hgt)
    plt.show()