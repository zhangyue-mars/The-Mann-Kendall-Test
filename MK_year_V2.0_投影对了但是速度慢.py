"""
@author:Zhang Yue
@date  :2023/2/10:21:41
@IDE   :PyCharm
"""
# coding:utf-8
import os
import sys
import numpy as np
import rasterio as ras
from tqdm import tqdm
from osgeo import gdal

#读取每个tiff图像的属性信息
def Readxy(RasterFile):
    ds = gdal.Open(RasterFile,gdal.GA_ReadOnly)
    if ds is None:
        print ('Cannot open ',RasterFile)
        sys.exit(1)
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    band = ds.GetRasterBand(1)
    # data = band.ReadAsArray(0,0,cols,rows)
    noDataValue = band.GetNoDataValue()
    projection=ds.GetProjection()
    geotransform = ds.GetGeoTransform()
    return rows,cols,geotransform,projection,noDataValue

# 写文件，写成tiff
def write_img(filename, im_proj, im_geotrans, im_data):
    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    # 判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
    dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    dataset.SetProjection(im_proj)  # 写入投影
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

def time_series_test(inputpath, outputPath):
    # inputpath:影像的存储路径
    # outputPath:影像的保存路径
    filepaths = []
    for file in os.listdir(inputpath):
        filepath1 = os.path.join(inputpath, file)
        filepaths.append(filepath1)
    # 获取影像数量
    num_images = len(filepaths)
    # 读取影像数据
    img1 = ras.open(filepaths[0])
    # 获取影像的投影，高度和宽度
    transform1 = img1.transform
    height1 = img1.height
    width1 = img1.width
    array1 = img1.read()
    img1.close()

    # 这一个没有参与运算，主要为了读取它的行列数、投影信息、坐标系和noData值
    rows, cols, geotransform, projection, noDataValue = Readxy(filepaths[0])
    print(rows, cols, geotransform, projection, noDataValue)

    # 读取所有影像
    for path1 in filepaths[1:]:
        if path1[-4:] == '.tif':
            print(path1)
            img2 = ras.open(path1)
            array2 = img2.read()
            array1 = np.vstack((array1, array2))
            img2.close()
    nums, width, height = array1.shape
    print(width, height)

    # 定义一个输出矩阵，可以将结果保存在此矩阵，无值区用-9999填充
    result = np.full([width, height], -9999.0000)
    result2 = np.full([width, height], -9999.0000)

    # 只有有值的区域才进行时间序列计算
    c1 = np.isnan(array1)
    sum_array1 = np.sum(c1, axis=0)
    nan_positions = np.where(sum_array1 == num_images)
    positions = np.where(sum_array1 != num_images)

    # 输出总像元数量
    print("all the pixel counts are {0}".format(len(positions[0])))

    # 时间序列运算
    for i in tqdm(range(len(positions[0]))):
    # for i in tqdm(range(15000132, 15000134)):
        # print(i)
        x = positions[0][i]
        y = positions[1][i]
        time_series_list = array1[:, x, y]

        # ******************时间序列运算开始的地方**********************#
        # 逐个时间序列进行计算，此处可以替换为需要的规则
        # 此处可以添加判断条件和运算规则
        a, b = calMutationalSite(time_series_list)
        # a = time_series_list[1]
        # b = time_series_list[2]
        # 将逐个时间序列运算的结果存在空的np数组中
        result[x, y] = a + 2000
        result2[x, y] = b + 2000
        # ******************时间序列运算结束的地方**********************#

    result_save_path = os.path.join(outputPath, "theFirstMutationalSite.tif")
    result2_save_path = os.path.join(outputPath, "theLastMutationalSite.tif")

    write_img(result_save_path, projection, geotransform, result)
    write_img(result2_save_path, projection, geotransform, result2)

def calMK(y):
    n = len(y)
    # 正序计算
    # 定义累计量序列Sk，长度n，初始值为0
    Sk = np.zeros(n)
    UFk = np.zeros(n)
    # 定义Sk序列元素s
    s = 0
    for i in range(1, n):
        for j in range(0, i):
            # if y.iloc[i] > y.iloc[j]:
            if y[i] > y[j]:
                s += 1
        Sk[i] = s
        E = (i + 1) * (i / 4)
        Var = (i + 1) * i * (2 * (i + 1) + 5) / 72
        UFk[i] = (Sk[i] - E) / np.sqrt(Var)
    # 逆序计算
    # 定义逆累计量序列Sk2
    y2 = np.zeros(n)
    Sk2 = np.zeros(n)
    UBk = np.zeros(n)
    s = 0
    y2 = y[::-1]
    for i in range(1, n):
        for j in range(0, i):
            # if y2.iloc[i] > y2.iloc[j]:
            if y2[i] > y2[j]:
                s += 1
        Sk2[i] = s
        E = (i + 1) * (i / 4)
        Var = (i + 1) * i * (2 * (i + 1) + 5) / 72
        UBk[i] = -(Sk2[i] - E) / np.sqrt(Var)

    UBk2 = UBk[::-1]
    return UFk, UBk2

def cross_point(line1, line2):
    point_is_exist = False
    x = y = 0
    x1,y1,x2,y2 = line1
    x3,y3,x4,y4 = line2
#     print("x1,y1,x2,y2:",x1,y1,x2,y2)
#     print("x3,y3,x4,y4:",x3,y3,x4,y4)
    if (x2 - x1) == 0:
        k1 = None
        b1 = 0
    else:
        k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
        b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键
    if (x4 - x3) == 0:  # L2直线斜率不存在
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k1 is None:
        if not k2 is None:
            x = x1
            y = k2 * x1 + b2
            point_is_exist = True
    elif k2 is None:
        x = x3
        y = k1 * x3 + b1
    elif not k2 == k1:
        x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0
        point_is_exist = True
    return point_is_exist, [x, y]

def calMutationalSite(y):
    mutationalSites = []
    UFk, UBk2 = calMK(y)
    # print(UFk, UBk2)
    nrows = len(y)
    #一行一行遍历excel中的数据：
    for i in range(1, nrows-1):
        test1 = UFk
        test2 = UBk2
        point1 = [i, test1[i], i + 1, test1[i + 1]]
        point2 = [i, test2[i], i + 1, test2[i + 1]]
        # 判断交点是否存在，坐标多少
        point_is_exist, [a, b] = cross_point(point1, point2)
        if a <= i + 1 and a >= i and point_is_exist == True:
            mutationalSites.append(int(a))
    return calFirstValue(mutationalSites), calLastValue(mutationalSites)

def calFirstValue(mutationalSites):
    if len(mutationalSites) == 0:
        mutationalSites = -9999
    else:
        mutationalSites = mutationalSites[0]
    return mutationalSites

def calLastValue(mutationalSites):
    if len(mutationalSites) == 0:
        mutationalSites1 = -9999
    else:
        mutationalSites1 = mutationalSites[len(mutationalSites)-1]
    return mutationalSites1

print("Start")
# 数据输入路径
input_path = r"E:\\RemoteSensing\\XYKH\\【python】MK突变检测\\inputdata\\vi\\"
# 结果数据保存路径
output_path = r"E:\\RemoteSensing\\XYKH\\【python】MK突变检测\\outputdata\\"
# 运行程序
time_series_test(input_path, output_path)
print("MK Finish")
