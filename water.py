#公众号：OpenCV与AI深度学习
#作者：Color Space
import cv2
import numpy as np
import random as rd

def watershed_algorithm(image):
    src = image.copy()
    # 边缘保留滤波EPF  去噪
    blur = cv2.pyrMeanShiftFiltering(image,sp=21,sr=55)
    # cv2.imshow("blur", blur)
    # 转成灰度图像
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    # 得到二值图像区间阈值
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow('thres image', binary)

    # 距离变换
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    dist_out = cv2.normalize(dist, 0, 1.0, cv2.NORM_MINMAX)
    # cv2.imshow('distance-Transform', dist_out * 100)
    ret, surface = cv2.threshold(dist_out, 0.5*dist_out.max(), 255, cv2.THRESH_BINARY)
    # cv2.imshow('surface', surface)
    sure_fg = np.uint8(surface)# 转成8位整型
    # cv2.imshow('Sure foreground', sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)  # 连通区域
    print(ret)
    markers = markers + 1 #整个图+1，使背景不是0而是1值

    # 未知区域标记(不能确定是前景还是背景)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel, iterations=1)
    unknown = binary - sure_fg
    # cv2.imshow('unknown',unknown)

    # 未知区域标记为0
    markers[unknown == 255] = 0
    # 区域标记结果
    markers_show = np.uint8(markers)
    # cv2.imshow('markers',markers_show*100)

    # 分水岭算法分割
    markers = cv2.watershed(image, markers=markers)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(markers)
    markers_8u = np.uint8(markers)

    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0),
              (255,0,255), (0,255,255), (255,128,0), (255,0,128),
              (128,255,0), (128,0,255), (255,128,128), (128,255,255)]
    for i in range(2,int(max_val+1)):
        ret, thres1 = cv2.threshold(markers_8u, i-1, 255, cv2.THRESH_BINARY)
        ret2, thres2 = cv2.threshold(markers_8u, i, 255, cv2.THRESH_BINARY)
        mask = thres1 - thres2
        # cv2.imshow('mask',mask)
        #color = (rd.randint(0,255), rd.randint(0,255), rd.randint(0,255))
        #image[markers == i] = [rd.randint(0,255), rd.randint(0,255), rd.randint(0,255)]
        #image[markers == i] = [colors[i-2]]
        
        contours,hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(image,contours,-1,colors[(i-2)%12],-1)
        #cv2.drawContours(src,contours,-1,colors[(i-2)%12],-1)
        M = cv2.moments(contours[0])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])#轮廓重心
        cv2.drawMarker(image, (cx,cy),(0,0,255),1,10,2)
        cv2.drawMarker(src, (cx,cy),(0,0,255),1,10,2)
        
    # cv2.putText(src,"count=%d"%(int(max_val-1)),(220,30),0,1,(0,255,0),2)
    # cv2.putText(image,"count=%d"%(int(max_val-1)),(220,30),0,1,(0,255,0),2)
    # cv2.imshow('regions', image)
    result = cv2.addWeighted(src,0.6,image,0.5,0) #图像权重叠加
    print('cnt:',int(max_val-1))
    cv2.imshow('result', result)

src = cv2.imread('./figure/Pearl.jpg', cv2.IMREAD_COLOR)
# cv2.imshow('src', src)
watershed_algorithm(src)
cv2.waitKey(0)
cv2.destroyAllWindows()
