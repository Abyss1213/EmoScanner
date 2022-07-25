"""
摄像头捕获
"""

import shutil
import os
import cv2
import time
from PyQt5.QtGui import QImage, QPixmap

from c3d_model import NUM_FRAMES_PER_CLIP, CROP_SIZE
save_dir = 'upload/images'
predict_list = 'upload/predict.list'
max_frame_num = 100
frame_rate = 30

def videoCapture(mainUI, src):
    if(os.path.exists(save_dir)):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    video_captor = cv2.VideoCapture(src)
    #video_captor = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    outfile = cv2.VideoWriter('video.avi',fourcc,frame_rate , (640,480),True)
    predictfile= open(predict_list, 'w')
    
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    face_cascade.load(r'haarcascade_frontalface_alt2.xml')  
        
    frames = []
    faces = []
    while(len(frames)<max_frame_num):
        ret, frame = video_captor.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = face_cascade.detectMultiScale(gray, 1.1, 5)
        if(mainUI.signal == 1):
            if(len(rects)):
                frames.append(frame)
                [x, y, w, h] = rects[0]
                face = frame[y+2:y+h,x+2:x+w]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                faces.append(face)
                
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face,(int(1.5*CROP_SIZE),int(1.5*CROP_SIZE)))
                qimg = QImage(face.data, face.shape[1], face.shape[0], QImage.Format_RGB888)
                mainUI.label_1_2.setPixmap(QPixmap.fromImage(qimg))
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        mainUI.label_1_1.setPixmap(QPixmap.fromImage(qimg))
        
        cv2.waitKey(10)
        if(mainUI.signal == -1):
            predictfile.close()
            video_captor.release()
            return
    mainUI.dialog.setVisible(False)
    batch_num = len(faces)//(NUM_FRAMES_PER_CLIP//2) -1;
    #batch_num = len(faces)//20;
    for frame in frames:
        outfile.write(frame)
    for i in range(batch_num):
        dir = save_dir + '/' + str(i)
        os.mkdir(dir)
        start_index = NUM_FRAMES_PER_CLIP//2 *i
        #start_index = 20 *i
        end_index = start_index + NUM_FRAMES_PER_CLIP
        for j in range(start_index, end_index):
            cv2.imwrite(os.path.join(dir,'%03d'%j+'.jpg'), faces[j])
        predictfile.write(dir +' 4\n')
    predictfile.close()
    video_captor.release()
              
if __name__ == '__main__':
    pass
    
  