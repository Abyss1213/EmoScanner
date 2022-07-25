"""
数据集预处理及各方法测试
"""
import os
import copy
import random
import shutil
import numpy as np
import cv2
import time
from sm4 import SM4Key
from PyQt5.QtGui import QImage, QPixmap

from c3d_model import NUM_FRAMES_PER_CLIP, CROP_SIZE
save_dir = 'images'
predict_list = 'list/predict.list'
max_frame_num = 100

casme2_dir = r'D:\share\CASME2\Cropped'
casme_dir = r'D:\share\CASME\Cropped'

all_file = 'list/filename_expression_label.txt'
train_file = 'list/train.txt'
test_file = 'list/test.txt'
test_casme_file = 'list/test_casme.txt'

EMOTIONS = ['happiness', 'disgust', 'repression', 'surprise','others']

def file2list(file, data_dir): 
    '''
    :txt转成绝对路径list
    '''
    lines = list(open(file, 'r'))
    fw = open(file.split('.')[0] + '.list','w')
    
    for line in lines:
        line = line.strip('\n').split('\t')
        if(len(line[0])==1):
            line[0] = '0' + line[0]
        path = os.path.join(data_dir, 'sub' + line[0], line[1])
        if(os.path.exists(path)):
            fw.write(path + ' ' + line[2] + ' ' + line[3] + '\n')
    fw.close()

def train_cut(i, train_list):
    '''
    :切分数据为train和valid
    '''
    lines = list(open(train_list))
    tw = open('list/train_part_'+str(i)+'.list','w')
    vw = open('list/valid_part_'+str(i)+'.list','w')
    
    dict = {}
    for line in lines:
        label = line.strip('\n').split()[-1]
        dict.setdefault(label, []).append(line)
    
    train_part = []
    valid_part = []
    for key in dict:
        lines = dict[key]
        random.shuffle(lines)
        one_five = len(lines)/5
        for j in range(int(i*one_five)):
            train_part.append(lines[j])
        for j in range(int((i+1)*one_five), len(lines)):
            train_part.append(lines[j])
        for j in range(int(i*one_five), int((i+1)*one_five)):
            valid_part.append(lines[j])
        
    for line in train_part:
        print(line)
        tw.write(line)
    for line in valid_part:
        print(line)
        vw.write(line)
    
    print('"train_cut" done ')
    tw.close()
    vw.close()
    
    return train_part, valid_part

def videoCut(save_dir, video_path, list = 'list/addition.list', label='4'):
    
    if(not os.path.exists(video_path)):
        print('cannot find video')
        return
    
    frames_dir = os.path.join(save_dir,'frames')
    faces_dir = os.path.join(save_dir,'faces')
    if(os.path.exists(frames_dir)):
        shutil.rmtree(frames_dir)
    os.mkdir(frames_dir)
    if(os.path.exists(faces_dir)):
        shutil.rmtree(faces_dir)
    os.mkdir(faces_dir)
    
    video_captor = cv2.VideoCapture(video_path)
    #video_captor = cv2.VideoCapture(0 + cv2.CAP_DSHOW)#去上下黑边
    file= open(list, 'w')
    
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    face_cascade.load(r'C:\Users\dyz110\AppData\Local\Programs\Python\Python37\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')  
        
    frames = []
    faces = []
    while(True):
        ret, frame = video_captor.read()
        if(not ret):
            break
        frame_cp = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = face_cascade.detectMultiScale(gray, 1.1, 5)
        if(len(rects)):
            frames.append(frame_cp)
            [x, y, w, h] = rects[0]
            face = frame[y+2:y+h,x+2:x+w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            faces.append(face)
        cv2.waitKey(10)
    NUM_FRAMES_PER_CLIP = 20
    batch_num = len(faces)//NUM_FRAMES_PER_CLIP;
    for i in range(batch_num):
        dir = os.path.join(frames_dir,str(i))
        os.mkdir(dir)
        start_index = NUM_FRAMES_PER_CLIP *i
        end_index = start_index + NUM_FRAMES_PER_CLIP
        for j in range(start_index, end_index):
            cv2.imwrite(os.path.join(dir,'%03d'%j+'.jpg'), frames[j])
        
    for i in range(batch_num):
        dir = os.path.join(faces_dir,str(i))
        os.mkdir(dir)
        start_index = NUM_FRAMES_PER_CLIP *i
        end_index = start_index + NUM_FRAMES_PER_CLIP
        for j in range(start_index, end_index):
            cv2.imwrite(os.path.join(dir,'%03d'%j+'.jpg'), faces[j])
        file.write(dir +' 4\n')
    file.close()
    video_captor.release()
   
def getlist(data_dir):
    addition_file= open('list/addition.list', 'w')
    for i in range(4):
        for j in range(7):
            addition_file.write(data_dir +'\\'+str(i)+'\\faces\\'+str(j)+ ' ' + str(i) +'\n')
    addition_file.close()

def compose(img_path):
    
    img = cv2.imread(img_path +'\\' + '%03d'%0 + '.jpg',1)
    imgInfo = img.shape
    size = (imgInfo[1],imgInfo[0])
    print(size)
    
    videoWrite = cv2.VideoWriter(img_path +'\\'+'show_video.mp4',-1,100,size)# 写入对象：1.fileName  2.-1：表示选择合适的编码器  3.视频的帧率  4.视频的size
    for i in range(100):
        fileName = img_path + '\\' + '%03d'%i + '.jpg'
        
        img = cv2.imread(fileName)
        videoWrite.write(img)# 写入方法  1.编码之前的图片数据
    print('end!')   

def rename_orderly(datadir):
    i = 0
    for root, dirs, files in os.walk(datadir):
        '''
        os.walk
        root:当前目录绝对路径
        dirs:当前目录下所有文件夹名列表
        files:当前目录下所有文件名列表
        '''
        for file in files:
            name, tail= file.split('.')
            os.rename(os.path.join(root, file), os.path.join(root, '%03d' % i + '.' +tail))
            i+=1

def free():
    save_dir = 'images'
    predict_list = 'predict.list'
    if(os.path.exists(save_dir)):
        shutil.rmtree(save_dir)
    if os.path.exists(predict_list):  
        os.remove(path)            

    

def check_arr_delete():
    arr = np.array([0,1,2,3,4])
    arr2 = np.delete(arr,2)
    print(arr2)

def check_sm4():
    key0 = SM4Key(b"###faceshield###")
    results = [[0.5, 0, 0.25, 0, 0], [0, 1, 0, 0, 0], [0.5, 0, 0.25, 0, 0], [0, 1, 0, 0, 0], [0.5, 0, 0.25, 0, 0],
               [0, 1, 0, 0, 0]]
    cipher = key0.encrypt(str(results).encode('utf-8'), initial=b"\0"*16,padding=True) 
    print(cipher)
    
    dec = key0.decrypt(cipher, initial=b"\0"*16,padding=True).decode('utf-8')
    print(dec)

import paramiko
import json
def check_code_file(array):
    with open("data.txt", "w", encoding='utf-8') as f:
        key0 = SM4Key(b"###faceshield###")
        cipher = bytes.decode(key0.encrypt(str(array).encode('utf-8'), initial=b"\0"*16,padding=True),"utf-8","ignore")
        print(type(cipher))
        '''
          1.消除乱码ensure_ascii=False
          2.把数据类型转换成字符串并存储在文件中
        '''
        b = json.dump(cipher, f, ensure_ascii=False)
        print('数据写入完毕！')


if __name__ == '__main__':
    #file2list(all_file,casme2_dir)
    #file2list(train_file,casme2_dir)
    #file2list(test_file, casme2_dir)
    #file2list(test_casme_file, casme_dir)
    #for i in range(5):
    #    train_cut(i, 'list/filename_expression_label.list')
#     print('video cut')
#     for i in [0,3]:
#         save_dir = r'D:\newFolder\other\xinan\dataset\addition' + '\\' +str(i)
#         videoCut(save_dir, save_dir + '\\' + str(i) +'.mp4')
#     print('done')
    #getlist(r'D:\newFolder\other\xinan\dataset\addition')
    #rename_orderly(r'D:\newFolder\other\xinan\dataset\addition\020')
    #compose(r'D:\newFolder\other\xinan\dataset\addition\020')
    #check_arr_delete()
    #check_sm4()
#     results = [[0.5, 0, 0.25, 0, 0], [0, 1, 0, 0, 0], [0.5, 0, 0.25, 0, 0], [0, 1, 0, 0, 0], [0.5, 0, 0.25, 0, 0],
#                [0, 1, 0, 0, 0]]
#     check_code_file(results)
    
    
