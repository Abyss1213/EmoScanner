"""
预测和展示
"""
import sys
import os
import time
import zipfile  
import paramiko
import json
from sm4 import SM4Key
import numpy as np
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from test_c3d_me import run_test
from upload_dir import *

EMOTIONS = ['愉悦', '厌恶', '压抑', '惊讶', '其他']
def predict_and_show(mainUI):
    '''
    预测和展示
    '''
    
    mainUI.label_1_3.setText("获取完成,正在分析......")
    QApplication.processEvents()
    
    #predict #######################################################
    
    #测试
    results_array, result_array = run_test("models/c3d_me_model-200", 'upload/predict.list', predict=True)
    mainUI.label_1_3.setText("分析完成")
    #print(results_array)

    #show #########################################################
    
    #mainUI.label_1_3.setText(str(results_array))
    shanxing_array = np.array(results_array).sum(axis=0)/len(results_array)
    zhexian_array = results_array
    print(zhexian_array)
    video_time = mainUI.label_1_1.video_time
    pie_and_line(shanxing_array, zhexian_array,video_time)
    mainUI.widget.view.load(QtCore.QUrl("file:///pie.html"))
    mainUI.widget_2.view.load(QtCore.QUrl("file:///second_line.html"))
    #shanxing(shanxing_array)
    #zhexian(zhexian_array,video_time)
    
    #mainUI.label_2_1.setPixmap(QtGui.QPixmap("result_pictures/pie_diagram.png"))
    #mainUI.label_3_1.setPixmap(QtGui.QPixmap("result_pictures/zhexian.png"))
    
    top,top2, abnormal = shanxingfenxijieguo(shanxing_array)
    mainUI.label_2_2_1.setText(EMOTIONS[top])
    mainUI.label_2_2_2.setText(EMOTIONS[top2])
    mainUI.label_2_2_3.setText(abnormal)
    mainUI.label_2_3.setText(fenxi_a(top))
    mainUI.label_2_4.setText(zonghefenxi(top,top2))
    mainUI.label_2_5.setText(tiwen_a(top))
    mainUI.label_2_6.setText(guancha_a(top))
    
    mainUI.label_3_2.setText(EMOTIONS[top])
    mainUI.label_3_3.setText(bodong(zhexian_array,video_time))
    mainUI.label_3_4.setText(fenxi_b(top))
    mainUI.label_3_5.setText(fenxibbodong(top))
    mainUI.label_3_6.setText(tiwen_b(top))
    mainUI.label_3_7.setText(guancha_b(top))


    if(top==0 and bodong(zhexian_array,video_time)=='无'):
        mainUI.label_3_3.setText(bodong(zhexian_array,video_time))
        mainUI.label_3_4.setText('情绪正常')
        mainUI.label_3_5.setText('情绪稳定，无波动')
        mainUI.label_3_6.setText('来自哪个国家\n在中国呆多久 ')
        mainUI.label_3_7.setText('是否有异常举动或特别行为')
        
    QApplication.processEvents()
    
    #en_results_array = sm4.
    #code_file(np.array(results_array).tolist())
    #res = sftp_upload_file()
    
    #上传
    upload()
    
    ##############################################################
from pyecharts.charts import Pie
from pyecharts.charts import Grid
from pyecharts.charts import Line, Page
from pyecharts import options as opts


def pie_and_line(fracs, shuzu, video_time):
    '''
    扇形图和折线图
    '''
    pie = (
        Pie(init_opts=opts.InitOpts(
                                width='512px',
                                height='384px',))
            .add("", [['happiness', "%.2f"%fracs[0]],
                      ['disgust', "%.2f"%fracs[1]],
                      ['repression', "%.2f"%fracs[2]],
                      ['surprise', "%.2f"%fracs[3]],
                      ['others', "%.2f"%fracs[4]]])  # 加入数据
            .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}")))  # 样式设置项

    # 定义一个Line_charts函数
    def Line_charts() -> Line:
        colors = ['orange', 'green', 'skyblue', 'tomato', 'black']
        colors = ['black', 'tomato', 'skyblue', 'lightgreen', 'orange']
        x = ['happiness', 'disgust', 'repression', 'surprise', 'others']
        t = ["%.2f"%(i*video_time/len(shuzu)) for i in range(0,len(shuzu))]
        print(len(t))
        c = Line(init_opts=opts.InitOpts(
                                width='512px',
                                height='384px'))
        c.add_xaxis(xaxis_data=t)
        for i in range(len(shuzu[0])):
            s = []
            for j in range(len(shuzu)):
                s.append("%.2f"%shuzu[j][i])
            c.add_yaxis(series_name=x[i], y_axis=s, color=colors[i], linestyle_opts=opts.LineStyleOpts(width=3))

        return c

    # 绘制图表
    c = Line_charts()
    # 生成到本地网页形式打开，也可自己设置保存成png图片，因为网页的使用更方便，自己按情况使用
    c.render("second_line.html")
    pie.render('pie.html')



def bodong(arr,video_time):#波动分析
    arr = np.array(arr)
    str = ['','']
    flag = [0,0]
    threshold1 = 0.75
    threshold2 = 0.5
    for i in range(len(arr)-1):
        tmp = arr[i+1] - arr[i]
        max = np.max(tmp)
        argmax = np.argmax(tmp)
        min = np.min(tmp)
        argmin = np.argmin(tmp)
        
        if(abs(max)>threshold1 or abs(min)>threshold1):
            if(flag[0]==0):
                str[0]+='极大波动：\n'
                flag[0]=1
            str[0]+="%.1f %s-->%s\n"%(i*video_time/len(arr),EMOTIONS[argmin],EMOTIONS[argmax])
        elif(abs(max)>threshold2 or abs(min)>threshold2):
            if(flag[1]==0):
                str[1]+='较大波动：\n'
                flag[1]=1
            str[1]+="%.1f %s-->%s\n"%(i*video_time/len(arr),EMOTIONS[argmin],EMOTIONS[argmax])
    result = str[0]+str[1]
    if(len(result)==0):
        result = "无"
    #print(result)
    return result

def shanxingfenxijieguo(arr):#扇形分析
    
    threshold = 0.5
    abnormal = "否"
    top = np.argmax(arr)
    if(arr[top]>threshold and top>0):
        abnormal = "是"
    arr[top]=0
    top2 = np.argmax(arr)
    if(arr[top2]>threshold and top>0):
        abnormal = "是"
    return top,top2, abnormal

def fenxi_a(emotion):  # 饼状图原因分析
    if emotion == 0:  # happiness
        return '内心需求得到满足,心情愉悦,对即将发生的情况有所掌握\n'
    elif emotion == 1:  # disgust 厌恶
        return '对所讨论话题讨厌,回避,想要尽快结束\n'
    elif emotion == 2:  #repression 克制
        return '故作镇定,掩饰自然情绪,掩饰自然生理反应\n'
    elif emotion == 3:  # surprise
        return '对发生的状况无法预料,对状况的预料失误,害怕出错\n'
    elif emotion == 4:  # others
        return '遇到危险,紧张,愤怒,受到挫折\n'


def fenxi_b(emotion):  # 折线图时段分析
    if emotion == 0:  #happiness
       return '达到目的,被信任,充满期待\n'
    elif emotion == 1:  #disgust
        return '极度焦虑或紧张而回避某些话题,因惧怕出错而试图逃避\n'
    elif emotion == 2:  #repression
        return '虚伪,极可能为了掩饰自然生理反应进而掩盖谎言\n'
    elif emotion == 3:  #surprise
        return '所处局面突然难以控制,惊讶过长则刻意为之\n'
    elif emotion == 4:  # otbers
        return '逃避,防御,惧怕,异常中立的表情常常刻意而为之,\n'

def fenxibbodong(emotion):#折线图波动分析
    if emotion == 0:
        return '真实的心情愉悦,抑或掩饰其他情绪\n'
    elif emotion == 1:
        return '被否定,需求被拒绝,对某事物憎恨\n'
    elif emotion == 2:  
        return '对发生的情况事先早有原料\n'
    elif emotion == 3:
        return '警惕是否引起他人怀疑,怀疑自己是否表露异常\n'
    elif emotion == 4:
        return '惧怕被发现,被怀疑,心理活动平静无异常\n'

def tiwen_a(emotion):#饼状图提问
    qlist = [[7, 9,10],  #happiness
             [6, 7,13],  #disgust
             [1, 2, 5, 9,11],  #repression
             [4, 7, 9,13],  #surprise
             [2, 3, 5, 8,12],  # others
             ]
    qdict = [
        '出/入境的目的是',
        '出/入境之后住哪',
        '在目的地有认识的朋友吗',
        '在目的地待多久',
        '有没有携带违禁品',
        '从事什么行业',
        '有需要申报的东西吗',
        '有携带毒品或易燃物吗',
        '是自己整理的行李吗',
        '目前的回答均属实吗',
        '有曾用名吗,有的话是什么',
        '有过特殊社会经历吗',
        '有其他人和你同行吗',
        '刚刚的回答有发现自己逻辑错误吗',

    ]

    text = ''
    for i in qlist[emotion]:
        text = text + (qdict[i - 1] + '?\n')
    return text


def tiwen_b(emotion):#折线图提问
    qlist = [[6, 8,11],  #happiness
             [8,9,12],  #disgust
             [3,4 ,10],  #repression
             [5, 6,12],  #surprise
             [ 6, 7,9,11],  # others
             ]
    qdict = [
        '出/入境的目的是',
        '出/入境之后住哪',
        '在目的地有认识的朋友吗',
        '在目的地待多久',
        '有没有携带违禁品',
        '从事什么行业',
        '有需要申报的东西吗',
        '有携带毒品或易燃物吗',
        '是自己整理的行李吗',
        '目前的回答均属实吗',
        '有曾用名吗,有的话是什么',
        '有过特殊社会经历吗',
        '有其他人和你同行吗',
        '刚刚的回答有发现自己逻辑错误吗',

    ]

    text = ''
    for i in qlist[emotion]:
        text = text + (qdict[i - 1] + '?\n')
    return text


def guancha_a(motion):#饼状图观察
    a = [''] * 5
    a[0] = '搬弄手指,回答时生硬的重复问题\n'
    a[1] = '轻微颤抖（压抑愤怒）,表情焦躁\n'
    a[2] = '频繁抿嘴,抬起下巴,眯眼睛,撇嘴,手放入口袋\n'
    a[3] = '无意识的搓手,玩弄头发\n'
    a[4] = '面色苍白,双手超过头顶的动作\n'
   
    if motion == 0:
        return a[0]
    if motion == 1:
        return a[1]
    if motion == 2:
        return a[2]
    if motion == 3:
        return a[3]
    if motion == 4:
        return a[4]
 
def guancha_b(motion):#折线图观察
    a = [''] * 5
    a[0] = '眼神躲闪,手放在脖颈后面\n'
    a[1] = '抖腿,蹭额头,摸鼻子,皱眉\n'
    a[2] = '单肩抖动,视线飘忽不定,频繁眨眼\n'
    a[3] = '长时间注视对方眼睛,右肩微耸,深呼吸\n'
    a[4] = '瞳孔放大,眉毛向上拉,手指交错\n'
   
    if motion == 0:
        return a[0]
    if motion == 1:
        return a[1]
    if motion == 2:
        return a[2]
    if motion == 3:
        return a[3]
    if motion == 4:
        return a[4]

def zonghefenxi(motion_a,motion_b):#综合分析
    a = [''] * 10  
    a[0] = '假装开心,实则讨厌话题\n'
    a[1] = '假装开心,实则故作镇定,压抑情绪\n'
    a[2] = '假装开心,实则掩饰恐慌\n'
    a[3] = '假装开心,实则压抑愤怒、悲伤、恐惧等其他情绪\n'
    a[4] = '隐瞒对话题的回避、反感\n'
    a[5] = '试图回避话题,却产生失误\n'
    a[6] = '试图回避话题时产生愤怒、沮丧、恐惧等情绪\n'
    a[7] = '试图隐藏惊讶之情,故作镇定\n'
    a[8] = '试图隐藏愤怒、悲伤、恐惧等情绪\n'
    a[9] = '对发生的结果难以预测而产生愤怒、恐惧等情绪\n'

    if motion_a== 0 and motion_b==1 or motion_a== 1 and motion_b==0:
        return a[0]
    if motion_a== 0 and motion_b==2 or motion_a== 2 and motion_b==0:
        return a[1]
    if motion_a== 0 and motion_b==3 or motion_a== 3 and motion_b==0:
        return a[2]
    if motion_a== 0 and motion_b==4 or motion_a== 4 and motion_b==0:
        return a[3]
    if motion_a== 1 and motion_b==2 or motion_a== 2 and motion_b==1:
        return a[4]
    if motion_a== 1 and motion_b==3 or motion_a== 3 and motion_b==1:
        return a[5]
    if motion_a== 1 and motion_b==4 or motion_a== 4 and motion_b==1:
        return a[6]
    if motion_a==2 and motion_b==3 or motion_a== 3 and motion_b==2:
        return a[7]
    if motion_a== 2 and motion_b==4 or motion_a== 4 and motion_b==2:
        return a[8]
    if motion_a== 3 and motion_b==4 or motion_a== 4 and motion_b==3:
        return a[9]
    return "正常"

