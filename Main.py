import sys
import cv2
import numpy as np
import pygame
from matplotlib import pyplot as plt
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5 import Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from ui.Ui_ps import Ui_MainWindow
from ui.Ui_child import Ui_Form
from ui.Ui_logon import Ui_LogonWindow
from ui.Ui_watermark import Ui_watermark
from ui.Ui_blur import Ui_blur
from ui.Ui_gradient import Ui_gradient
from ui.Ui_contours import Ui_contours

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        #设置图标
        icon = QIcon()
        icon.addPixmap(QPixmap('hunt.png'))
        self.setWindowIcon(icon)

        self.img = None
        self.fname = None
        self.dict = {}#存储执行操作 用于撤回
        self.dict_2 = {}#存储恢复操作
        self.index = 0
        self.num = 0
    #设立显示窗口的函数
    def Open(self):
        self.show()
    
    #将按键和函数连接起来
    def connection(self):
        self.actionwenjian.triggered.connect(self.openFile)#打开文件
        self.action.triggered.connect(self.saveImage)#图片另存为
        self.action_2.triggered.connect(self.withdraw)#撤销
        self.action_3.triggered.connect(self.restore)#还原
        self.pushButton_10.clicked.connect(self.left_rotate)#左旋转
        self.pushButton_11.clicked.connect(self.right_rotate)#右旋转
        self.pushButton.clicked.connect(self.grayscale)#灰度图
        self.pushButton_3.clicked.connect(self.orb)#特征点检测
        # self.pushButton_4.clicked.connect(self.sobelOperter)#图像梯度
        self.pushButton_8.clicked.connect(self.original_map)#显示原图
        # self.pushButton_5.clicked.connect(self.find_Contours)#图像轮廓
        self.pushButton_7.clicked.connect(self.grab_cut)#抠图
        self.pushButton_12.clicked.connect(self.image_hist)#直方图

    #图片自适应大小
    def image_size(self,img):
        width = img.width()  #获取图片宽度
        height = img.height() #获取图片高度
        if width / self.label.width() >= height / self.label.height(): #比较图片宽度与label宽度之比和图片高度与label高度之比
            ratio = width / self.label.width()
        else:
            ratio = height / self.label.height()
        new_width = width / ratio  #定义新图片的宽和高
        new_height = height / ratio
        return new_width,new_height
    
    #图片备份方便撤回和复原
    def image_copy(self):
        self.img_copy = self.img[:]#备份图像
        self.index += 1
        self.dict[self.index] = self.img_copy
    #图片显示
    def image_show(self):
        image = QImage(self.img[:], self.img.shape[1], self.img.shape[0], self.img.shape[1] * 3, QImage.Format_BGR888)
        width = image.width()  #获取图片宽度
        height = image.height() #获取图片高度
        if width / self.label.width() >= height / self.label.height(): #比较图片宽度与label宽度之比和图片高度与label高度之比
            ratio = width / self.label.width()
        else:
            ratio = height / self.label.height()
        new_width = width / ratio  #定义新图片的宽和高
        new_height = height / ratio
        # 将图片转换为QPixmap方便显示
        image = QPixmap.fromImage(image).scaled(new_width, new_height)
        # 使用label进行显示
        self.label.setPixmap(image)

    #撤回
    def withdraw(self):
        if self.img is None:
            return
        if self.dict is None:
            return
        if self.index<=1:#当key值为1时则不进行撤回操作
            return
        self.num += 1
        self.dict_2[self.num] = self.dict[self.index]#撤回的步骤备份

        del self.dict[self.index]
        self.index = self.index-1

        self.img =self.dict[self.index]
        img_value = QImage(self.img[:], self.img.shape[1], self.img.shape[0], self.img.shape[1] * 3, QImage.Format_BGR888)
        new_width,new_height = self.image_size(img_value)
        img_value = QPixmap.fromImage(img_value).scaled(new_width, new_height)
        self.label.setPixmap(img_value)

        print("撤回字典长度为:",len(self.dict))
        
    
    #还原功能
    def restore(self):
        if self.img is None:
            return
        if self.dict is None:
            return
        if self.num<=0:#当key值为0时则不进行还原操作
            return
        self.index += 1
        self.dict[self.index] = self.dict_2[self.num]

        self.img =self.dict_2[self.num]
        img_value = QImage(self.img[:], self.img.shape[1], self.img.shape[0], self.img.shape[1] * 3, QImage.Format_BGR888)
        new_width,new_height = self.image_size(img_value)
        img_value = QPixmap.fromImage(img_value).scaled(new_width, new_height)
        self.label.setPixmap(img_value)

        del self.dict_2[self.num]
        self.num = self.num-1
        print("还原字典长度为：",len(self.dict_2))

    #打开(O)文件
    def openFile(self):
        self.fname,type = QFileDialog.getOpenFileName(self, '打开文件', '.', '图像文件(*.jpg *.png)')
        try:
            #防止中文乱码
            self.img = cv2.imdecode(np.fromfile(self.fname, dtype=np.uint8), cv2.IMREAD_COLOR)
            #图片备份
            self.image_copy()
            #图片显示
            self.image_show()
        except:
            print("未读取到文件！")

        # self.label.setPixmap(QPixmap(self.fname))    #原图大小显示
        # png = QPixmap(self.fname).scaled(self.label.width(), self.label.height())  #按窗口大小显示
        # self.label.setPixmap(png)

    #另存为图片
    def saveImage(self):  
        if self.img is None:
            return
        fd,type= QFileDialog.getSaveFileName(self, "保存图片", "", "*.jpg;;*.png;;All Files(*)")
        png = QImage(self.img[:], self.img.shape[1], self.img.shape[0], self.img.shape[1] * 3, QImage.Format_BGR888)
        png.save(fd)

        

    #左旋转
    def left_rotate(self):
        if self.img is None:
            return
        self.img = self.dict[len(self.dict)]
        
        self.img = cv2.rotate(self.img,cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        #图片备份
        self.image_copy()
        #图片显示
        self.image_show()
    #右旋转
    def right_rotate(self):
        if self.img is None:
            return
        self.img = self.dict[len(self.dict)]
        self.img = cv2.rotate(self.img,cv2.ROTATE_90_CLOCKWISE)
        
        #图片备份
        self.image_copy()
        #图片显示
        self.image_show()
 
    #显示原图
    def original_map(self):
        if self.img is None:
            return
        self.img = cv2.imread(self.fname)
        
        #图片备份
        self.image_copy()
        #图片显示
        self.image_show()
    #灰度图
    def grayscale(self):
        if self.img is None:
            return
        self.img = self.dict[len(self.dict)]
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        
        #图片备份
        self.image_copy()
        #图片显示
        self.image_show()
    
    #平滑处理
    def gaussianBlur(self,input_blur,input_resize):
        if self.img is None:
            return
        self.img = self.dict[len(self.dict)]
        #核大小
        if input_resize == '(1,1)':
            input_true_resize = (1,1)
        elif input_resize == '(3,3)':
            input_true_resize = (3,3)
        elif input_resize == '(5,5)':
            input_true_resize = (5,5)
        elif input_resize == '(7,7)':
            input_true_resize = (7,7)
        elif input_resize == '(9,9)':
            input_true_resize = (9,9)
        elif input_resize == '(11,11)':
            input_true_resize = (11,11)
        elif input_resize == '(13,13)':
            input_true_resize = (13,13)
        elif input_resize == '(15,15)':
            input_true_resize = (15,15)
        elif input_resize == '(17,17)':
            input_true_resize = (17,17)       
        elif input_resize == '(19,19)':
            input_true_resize = (19,19)                    
        #模糊类型
        if input_blur == '高斯模糊':
            self.img = cv2.GaussianBlur(self.img,input_true_resize,0,0)
        elif input_blur == '均值模糊':
            self.img = cv2.blur(self.img,input_true_resize)
        elif input_blur == '方框模糊':
            self.img = cv2.boxFilter(self.img,-1,input_true_resize)
        
        #图片备份
        self.image_copy()
        #图片显示
        self.image_show()

    #特征点检测
    def orb(self):
        if self.img is None:
            return
        self.img = self.dict[len(self.dict)]
        #创建orb检测器
        orb = cv2.ORB_create()
        kps = orb.detect(self.img)
        #-1表示随机颜色
        self.img = cv2.drawKeypoints(self.img, kps, None, -1, cv2.DrawMatchesFlags_DEFAULT)  
        
        #图片备份
        self.image_copy()
        #图片显示
        self.image_show()

    #图像梯度
    def sobelOperter(self,input_operter,input_ksize):
        if self.img is None:
            return
        self.img = self.dict[len(self.dict)]
        if input_operter == 'Sobel':
            #sobel算子
            Sobelx = cv2.Sobel(self.img,cv2.CV_64F,1,0,ksize=input_ksize)
            Sobely = cv2.Sobel(self.img,cv2.CV_64F,0,1,ksize=input_ksize)
            Sobelx = cv2.convertScaleAbs(Sobelx)
            Sobely = cv2.convertScaleAbs(Sobely)
            self.img = cv2.addWeighted(Sobelx,0.5,Sobely,0.5,0)
        elif input_operter == 'Scharr':
            input_ksize = -1
            Sobelx = cv2.Sobel(self.img,cv2.CV_64F,1,0,ksize=input_ksize)
            Sobely = cv2.Sobel(self.img,cv2.CV_64F,0,1,ksize=input_ksize)
            Sobelx = cv2.convertScaleAbs(Sobelx)
            Sobely = cv2.convertScaleAbs(Sobely)
            self.img = cv2.addWeighted(Sobelx,0.5,Sobely,0.5,0)
        elif input_operter == 'Laplacian':
            input_ksize = None
            Laplacian = cv2.Laplacian(self.img,cv2.CV_64F)
            self.img = cv2.convertScaleAbs(Laplacian)
        #图片备份
        self.image_copy()
        #图片显示
        self.image_show()

    #轮廓检测
    def find_Contours(self,input_color,input_thickness):
        if self.img is None:
            return
        self.img = self.dict[len(self.dict)]
        #判断颜色
        if input_color == '红色':
            input_true_color = (0,0,255)
        elif input_color == '蓝色':
            input_true_color =(255,0,0)
        elif input_color == '绿色':
            input_true_color =(0,255,0)
        elif input_color == '黄色':
            input_true_color =(0,255,255)
        elif input_color == '紫色':
            input_true_color =(255,0,255)
        elif input_color == '青色':
            input_true_color =(255,255,0)
        elif input_color == '黑色':
            input_true_color =(0,0,0)
        elif input_color == '白色':
            input_true_color =(255,255,255)
        #对原图进行二值化处理
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        '''
        查找轮廓
        contours返回轮廓坐标
        hierarchy返回轮廓的组织结构
        '''
        print("-----------------------------")
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # img_contours = self.img
        #绘制轮廓
        self.img = cv2.drawContours(self.img, contours, -1, input_true_color, input_thickness) #img为三通道才能
        print(self.img)
        #图片备份
        self.image_copy()
        #图片显示
        self.image_show()

    #腐蚀
    def image_erode(self):
        if self.img is None:
            return
        self.img = self.dict[len(self.dict)]
        #滤波器的尺寸越大，效果覆盖范围越大
        kernel = np.ones((3,3), dtype=np.uint8)
        self.img = cv2.erode(self.img, kernel, iterations=1)
        #图片备份
        self.image_copy()
        #图片显示
        self.image_show()
    #膨胀
    def image_dilate(self):
        if self.img is None:
            return
        self.img = self.dict[len(self.dict)]
        #滤波器的尺寸越大，效果覆盖范围越大
        kernel = np.ones((3,3), dtype=np.uint8)
        self.img = cv2.dilate(self.img, kernel, iterations=1)
        #图片备份
        self.image_copy()
        #图片显示
        self.image_show()
    #礼帽运算
    def top_hat(self):
        if self.img is None:
            return
        self.img = self.dict[len(self.dict)]
        k = np.ones((3,3), np.uint8)
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_TOPHAT, k)
        #图片备份
        self.image_copy()
        #图片显示
        self.image_show()
    #黑帽运算
    def black_hat(self):
        if self.img is None:
            return
        self.img = self.dict[len(self.dict)]
        k = np.ones((3,3), np.uint8)
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_BLACKHAT, k)
        #图片备份
        self.image_copy()
        #图片显示
        self.image_show()

    #抠图
    def grab_cut(self):
        if self.img is None:
            return
        self.img = self.dict[len(self.dict)]
        self.img = cv2.resize(self.img, (0,0), fx=0.5, fy=0.5)
        try:
            r = cv2.selectROI('input', self.img, False)  # 返回 (x_min, y_min, w, h)
            # roi区域
            roi = self.img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]#高,宽
            img = self.img.copy()
            cv2.rectangle(img, (int(r[0]), int(r[1])),(int(r[0])+int(r[2]), int(r[1])+ int(r[3])), (0, 0, 255), 2)
            # 原图mask
            mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
            # 矩形roi
            rect = (int(r[0]), int(r[1]), int(r[2]), int(r[3])) # 包括前景的矩形，格式为(x,y,w,h)

            bgdmodel = np.zeros((1,65),np.float64) # bg模型的临时数组  13 * iterCount
            fgdmodel = np.zeros((1,65),np.float64) # fg模型的临时数组  13 * iterCount

            cv2.grabCut(self.img,mask,rect,bgdmodel,fgdmodel, 11, mode=cv2.GC_INIT_WITH_RECT)
            '''
            - img --> 输入的三通道图像；
            - mask --> 输入的单通道图像，初始化方式为GC_INIT_WITH_RECT表示ROI区域可以被初始化为：
                - GC_BGD --> 定义为明显的背景像素 0
                - GC_FGD --> 定义为明显的前景像素 1
                - GC_PR_BGD --> 定义为可能的背景像素 2
                - GC_PR_FGD --> 定义为可能的前景像素 3
            - rect --> 表示roi区域；
            - bgdModel --> 表示临时背景模型数组；
            - fgdModel --> 表示临时前景模型数组；
            - iterCount --> 表示图割算法迭代次数, 次数越多，效果越好；
            - mode --> 当使用用户提供的roi时候使用GC_INIT_WITH_RECT。
            '''
            # 提取前景和可能的前景区域
            mask2 = np.where((mask==1) + (mask==3), 255, 0).astype('uint8')
            print(mask2.shape)
            self.img = cv2.bitwise_and(self.img,self.img,mask=mask2)
            #图片备份
            self.image_copy()
            #图片显示
            self.image_show()
        except:
            print("结束操作")
    #添加水印
    def watermark(self,input_text,input_fontFace,input_fontScale,input_color,input_thickness):
        if self.img is None:
            return
        self.img = self.dict[len(self.dict)]
        blank_img = np.zeros(shape=(self.img.shape[0],self.img.shape[1],3), dtype=np.uint8)
        #判断颜色
        if input_color == '红色':
            input_true_color = (0,0,255)
        elif input_color == '蓝色':
            input_true_color =(255,0,0)
        elif input_color == '绿色':
            input_true_color =(0,255,0)
        elif input_color == '黄色':
            input_true_color =(0,255,255)
        elif input_color == '紫色':
            input_true_color =(255,0,255)
        elif input_color == '青色':
            input_true_color =(255,255,0)
        elif input_color == '黑色':
            input_true_color =(0,0,0)
        elif input_color == '白色':
            input_true_color =(255,255,255)
        #检测字体类型
        if input_fontFace == 'FONT_HERSHEY_COMPLEX':
            input_true_fontFace = cv2.FONT_HERSHEY_COMPLEX
        if input_fontFace == 'FONT_HERSHEY_COMPLEX_SMALL':
            input_true_fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL
        if input_fontFace == 'FONT_HERSHEY_DUPLEX':
            input_true_fontFace = cv2.FONT_HERSHEY_DUPLEX
        if input_fontFace == 'FONT_HERSHEY_PLAIN':
            input_true_fontFace = cv2.FONT_HERSHEY_PLAIN
        if input_fontFace == 'FONT_HERSHEY_SCRIPT_COMPLEX':
            input_true_fontFace = cv2.FONT_HERSHEY_SCRIPT_COMPLEX    
        if input_fontFace == 'FONT_HERSHEY_SCRIPT_SIMPLEX':
            input_true_fontFace = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        if input_fontFace == 'FONT_HERSHEY_SIMPLEX':
            input_true_fontFace = cv2.FONT_HERSHEY_SIMPLEX
        if input_fontFace == 'FONT_HERSHEY_TRIPLEX':
            input_true_fontFace = cv2.FONT_HERSHEY_TRIPLEX          
        #图片 文字 位置 字体类型 字体大小 字体颜色 字体粗细 行的类型
        cv2.putText(blank_img,text=input_text,org=(0, 400),
            fontFace=input_true_fontFace,fontScale=input_fontScale,
            color=input_true_color,thickness=input_thickness,lineType=cv2.LINE_4)
        #将水印和图片结合在一起
	    #参数分别为：图1，图1的权重，图2，图2的权重，权重和添加的值为3
        self.img = cv2.addWeighted(src1=self.img, alpha=1,src2=blank_img, beta=1, gamma = 2)
        #图片备份
        self.image_copy()
        #图片显示
        self.image_show()

    #彩色直方图
    def image_hist(self):
        if self.img is None:
            return
        color = ('blue', 'green', 'red')
        for i, color in enumerate(color):
            hist = cv2.calcHist([self.img], [i], None, [256], [0, 256])
            plt.plot(hist, color = color)
            plt.xlim([0, 256])
        plt.show()
#登录类
class LogonWindow(QtWidgets.QMainWindow, Ui_LogonWindow):
    def __init__(self):
        super(LogonWindow,self).__init__()
        self.setupUi(self)

        #设置图标
        icon = QIcon()
        icon.addPixmap(QPixmap('hunt.png'))
        self.setWindowIcon(icon)

        self.show()
        #播放音乐
        pygame.mixer.init()
        pygame.mixer.music.load("robot.mp3")
        pygame.mixer.music.play()
    def Close(self):
        pygame.mixer.quit()

#平滑处理类   
class BlurWindow(QtWidgets.QMainWindow, Ui_blur):
    def __init__(self):
        super(BlurWindow,self).__init__()
        self.setupUi(self)
        #设置图标
        icon = QIcon()
        icon.addPixmap(QPixmap('hunt.png'))
        self.setWindowIcon(icon)
    #设立显示窗口的函数
    def Open(self):
        self.show()
#图像梯度类   
class GradientWindow(QtWidgets.QMainWindow, Ui_gradient):
    def __init__(self):
        super(GradientWindow,self).__init__()
        self.setupUi(self)
        #设置图标
        icon = QIcon()
        icon.addPixmap(QPixmap('hunt.png'))
        self.setWindowIcon(icon)
    #设立显示窗口的函数
    def Open(self):
        self.show()
#图像轮廓类
class ContoursWindow(QtWidgets.QMainWindow, Ui_contours):
    def __init__(self):
        super(ContoursWindow,self).__init__()
        self.setupUi(self)
        #设置图标
        icon = QIcon()
        icon.addPixmap(QPixmap('hunt.png'))
        self.setWindowIcon(icon)
    #设立显示窗口的函数
    def Open(self):
        self.show()
#形态学类
class ChildWindow(QtWidgets.QMainWindow, Ui_Form):
    def __init__(self):
        super(ChildWindow,self).__init__()
        self.setupUi(self)
        #设置图标
        icon = QIcon()
        icon.addPixmap(QPixmap('hunt.png'))
        self.setWindowIcon(icon)
    #设立显示窗口的函数
    def Open(self):
        self.show()

class WaterWindow(QtWidgets.QMainWindow, Ui_watermark):
    def __init__(self):
        super(WaterWindow,self).__init__()
        self.setupUi(self)
        #设置图标
        icon = QIcon()
        icon.addPixmap(QPixmap('hunt.png'))
        self.setWindowIcon(icon)
    #设立显示窗口的函数
    def Open(self):
        self.show()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    #开始窗口
    logon = LogonWindow()
    main = MainWindow()
    main.connection()
    logon.pushButton.clicked.connect(logon.Close)#点击开始时音频停止
    logon.pushButton.clicked.connect(main.Open)#进行登录操作
    #添加平滑处理子窗口
    blur = BlurWindow()
    main.pushButton_2.clicked.connect(blur.Open)#子窗口显示
    blur.pushButton.clicked.connect(lambda:main.gaussianBlur(blur.comboBox_2.currentText(),blur.comboBox.currentText()))#完成选择

    #添加图像梯度子窗口
    gradient = GradientWindow()
    main.pushButton_4.clicked.connect(gradient.Open)#子窗口显示
    gradient.pushButton.clicked.connect(lambda:main.sobelOperter(gradient.comboBox_2.currentText(),gradient.spinBox.value()))#完成选择

    #添加图像轮廓子窗口
    contours = ContoursWindow()
    main.pushButton_5.clicked.connect(contours.Open)#子窗口显示
    contours.pushButton.clicked.connect(lambda:main.find_Contours(contours.comboBox.currentText(),contours.spinBox_2.value()))#完成选择

    
    #形态学子窗口
    child = ChildWindow()
    main.pushButton_6.clicked.connect(child.Open)#子窗口显示
    child.pushButton.clicked.connect(main.image_erode)#腐蚀
    child.pushButton_2.clicked.connect(main.image_dilate)#膨胀
    child.pushButton_3.clicked.connect(main.top_hat)#礼帽运算
    child.pushButton_4.clicked.connect(main.black_hat)#黑帽运算

    #添加水印子窗口
    water = WaterWindow()
    main.pushButton_9.clicked.connect(water.Open)#子窗口显示
    #水印内容   字体类型   字体大小  字体颜色  字体粗细
    water.pushButton.clicked.connect(lambda:main.watermark(water.lineEdit.text()
    ,water.comboBox_2.currentText(),water.spinBox.value()
    ,water.comboBox.currentText(),water.spinBox_2.value()))#完成确定
    sys.exit(app.exec_())
