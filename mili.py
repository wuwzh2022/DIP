import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
 
def show(img,name):
    cv2.namedWindow(name, 2)   
    cv2.imshow(name, img) 
 
def rice_area(img):
    img = cv2.imread(img)
    # 转换为灰度图
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 使用自适应阈值操作进行图像二值化
    dst = cv2.adaptiveThreshold(gray,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,101, 1)
    # 形态学去噪
    element = cv2.getStructuringElement(cv2.MORPH_RECT,(2, 2))
    # 开运算去噪（先腐蚀再膨胀）
    dst=cv2.morphologyEx(dst,cv2.MORPH_OPEN,element,iterations=2)
    kernel = np.ones((5,5),np.uint8)
    #dst = cv2.erode(dst, kernel, iterations = 1)
    
    # 轮廓检测函数
    contours, hierarchy = cv2.findContours(dst,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 100:
            # 创建一个全黑的mask
            mask = np.zeros_like(dst)

            # 在mask上绘制轮廓
            cv2.drawContours(mask, [contour], -1, (255,255,255), thickness=cv2.FILLED)

            # 对原图像中对应的部分进行腐蚀操作
            eroded = cv2.erode(dst, kernel, iterations=1)
            dst = cv2.bitwise_and(dst, dst, mask=cv2.bitwise_not(mask))
            eroded_masked = cv2.bitwise_and(eroded, eroded, mask=mask)
            dst = cv2.bitwise_or(dst, eroded_masked)

    contours, hierarchy = cv2.findContours(dst,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(dst,contours,-1,(120,0,0),2)
 
    count=0
    ares_avrg=0  
    # 遍历找到的所有米粒
    for cont in contours:
        # 计算包围性状的面积
        ares = cv2.contourArea(cont)
        # 过滤面积小于50的形状
        if ares<1:   
            continue
        count+=1
        ares_avrg+=ares
        # 打印出每个米粒的面积
        print("{}-blob:{}".format(count,ares),end="  ") 
        # 提取矩形坐标
        rect = cv2.boundingRect(cont) 
        print("x:{} y:{}".format(rect[0],rect[1]))
        # 绘制矩形
        cv2.rectangle(img,rect,(0,0,255),1)
        # 防止编号到图片之外（上面）,因为绘制编号写在左上角，所以让最上面的米粒的y小于10的变为10个像素
        y=10 if rect[1]<10 else rect[1] 
        # 在米粒左上角写上编号
        cv2.putText(img,str(count), (rect[0], y), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1) 
    print('个数',count,' 总面积',ares_avrg,' ares',ares)
    print("米粒平均面积:{}".format(round(ares_avrg/count,2))) 
 
    cv2.namedWindow("imgshow", 2) 
    cv2.imshow('imgshow', img)  
 
    cv2.namedWindow("dst", 2)   
    cv2.imshow("dst", dst)  
 
    cv2.waitKey()
 

rice_area('./figure/Rice.jpg')