self.lab = QLabel("标签字体大小颜色", self)
self.lab.setGeometry(50,50,300,200)        
self.setStyleSheet("QLabel{color:rgb(225,22,173,255);font-size:50px;font-weight:normal;font-family:Arial;}")

color:rgb()中的四个参数,前三个是控制颜色,第四个控制透明度
font-size:设置字体大小
font-weight:bold可设置字体加粗
font-family:选择自己想要的颜色
setStyleSheet同时可以设置标签背景图片,但无法使图片与标签大小匹配

self.label_2.setStyleSheet("QLabel{color:bgr(0,0,0,255);font-size:15px;font-weight:bold;font-family:Arial;}")

# setStyleSheet()
## 样式-英文	                          参数	                                   样式-中文
color:	                               white rgb(110,110,110) #eb7350	          前景颜色，字体颜色
background:	                          transparent	                              背景为透明
background-color:	                     white rgb(110,110,110) #eb7350	          背景颜色
background-position:	                eft right center top bottom	               设定图片的位置
background-image:url()	                ./img/back.jpg	                         背景图片 ，不缩放图片大小
border-image:url()	                     ./img/back.jpg	                         背景图片，会对图片进行拉伸，平铺
border-style:	                          outset inset	                              边框样式，按下是inset
border-width:	                          px	                                        边框大小
border-radius:	                          px	                                        边框弧度
border:3px solid red	                px color	                                   边框宽度以及颜色
border-color: rgba(255, 225, 255, 30);	 color	                                   边框颜色
font-family	                          微软雅黑	                                    设定字体所属家族
font: bold 14px	                     bold px	                                   字体大小并加粗
font-size:	                          px	                                        字体大小
font-style:	                          inset	                                   字体样式
font-weight:	                          px	                                        字体深浅
selection-color:	                     color	                                   设定选中时候的颜色


2,撤回 恢复操作
创建一个字典 和 index

每执行一次操作就把copy的图片添加到字典里面且索引++

撤回时：通过（索引-1）找到之前的图片并且显示他

恢复时：通过(索引+1）找到后面的图片并且显示他

3.图片在label居中显示
self.label.setAlignment(QtCore.Qt.AlignCenter)

#隐藏边框
        LogonWindow.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        LogonWindow.setAttribute(QtCore.Qt.WA_TranslucentBackground)


#按钮样式表
#pushButton{
	background-color:rgb(0,0,0);
	color:rgb(255,255,255);
	border:3px solid rgb(0,0,0);
	border-radius:10px
}
#pushButton:hover{
	background-color:rgb(255,255,255);
	color:rgb(0,0,0);
}
#pushButton:pressed{
	padding-top:5px;
	padding-left:5px;
}
#快捷键设置
self.action_2.setShortcut('ctrl+z')