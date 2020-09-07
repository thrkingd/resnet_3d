import os
import glob
import cv2
home_path = os.getcwd()
jpg_path = os.path.join(home_path,'work','data','ucf101')
video_path = os.path.join(home_path,'data','data48916','UCF-101')
video_class_list = glob.glob(video_path+'/*')
#handstandingpushup这个视频要把testlist中的stand改成Stand，avi的名字跟testlist有一点点区别，特别坑爹
n=0
all_video=0
for item in video_class_list:
    n=n+1
    
    video_list = glob.glob(item+'/*.avi')
    print(item.split('/')[-1],n)
    for avi in video_list:
        print(avi)
        name = avi.split('/')[-1]
        name = name.split('.')[0]
        jpg_folder = os.path.join(jpg_path,name)
        print(jpg_folder)
        if(not os.path.exists(jpg_folder)):
            os.mkdir(jpg_folder)
        cap = cv2.VideoCapture(avi)
        No = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if(ret is False):
                break
            No = No+1
            cv2.imwrite(os.path.join(jpg_folder,'frame{:06d}.jpg'.format(No)),frame)










