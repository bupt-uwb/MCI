import argparse
from time import sleep
from unicodedata import name
import numpy as np
import os
import threading
import argparse
import cv2
import tools
from PIL import Image, ImageDraw, ImageFont
from uwb_radar_with_phase import uwb_radar_with_phase
from human_detection import human_detector
from socket import *
import time
import select

'''
用于传递YOLO v5需要的参数，部分参数已设置默认值
'''
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='.\\yolov5s.pt', help='model path(s)')                                                                            
    parser.add_argument('--source', type=str, default='1', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', default=True, help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, default=[0], help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=True, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true',                                                 help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # print_args(FILE.stem, opt)
    return opt

def draw():
    global hd, uwb, thermal_frame, thermal_image, end, smaller_k, phase_flag
    hr_list = [[80, 80, 80, 80, 80]]  # 二维
    br_list = [[80, 80, 80, 80, 80]]  # 二维
    t_list = np.zeros(1)  # 一维
    num = 1
    last_num = num
    while 1:
        if hd.work:
            isman_camera = hd.isman_camera
            num = hd.num
            iou = np.array(hd.iou)
            iou_sorted = iou.copy()
            frame = hd.frame_output
            t_frame = thermal_frame.copy()
            cam_width = frame.shape[1]
            cam_height = frame.shape[0]
            thermal_width = 384
            thermal_height = 288

            if isman_camera and uwb.can_read:
                amplitude, index, energy_ratio = tools.sp(uwb.pure_data[-1, :], num)
                temp = []

                if num <= last_num:
                    hr_list = hr_list[0: num: 1]
                    br_list = br_list[0: num: 1]
                    for i in range(num):
                        d = (index[i] + 50) / 156 + 0.2
                        h1, b1, k_val1, h2, b2, k_val2 = uwb.vital_signs(d - 0.1, d + 0.1)
                        if k_val1 or k_val2:
                            if k_val1 <= k_val2:
                                smaller_k = k_val1
                                hr_list[i].append(int(round(h1)))
                                br_list[i].append(int(round(b1)))
                                if len(hr_list[i]) > 5:
                                    hr_list[i].pop(0)
                                if len(br_list[i]) > 5:
                                    br_list[i].pop(0)
                            else:
                                smaller_k = k_val2
                                phase_flag = 1
                                hr_list[i].append(int(round(h2)))
                                br_list[i].append(int(round(b2)))
                                if len(hr_list[i]) > 5:
                                    hr_list[i].pop(0)
                                if len(br_list[i]) > 5:
                                    br_list[i].pop(0)

                elif num > last_num:
                    for i in range(last_num):
                        d = (index[i] + 50) / 156 + 0.2
                        h1, b1, k_val1, h2, b2, k_val2 = uwb.vital_signs(d - 0.1, d + 0.1)
                        if k_val1 or k_val2:
                            if k_val1 <= k_val2:
                                smaller_k = k_val1
                                hr_list[i].append(int(round(h1)))
                                br_list[i].append(int(round(b1)))
                                if len(hr_list[i]) > 5:
                                    hr_list[i].pop(0)
                                if len(br_list[i]) > 5:
                                    br_list[i].pop(0)
                            else:
                                smaller_k = k_val2
                                phase_flag = 1
                                hr_list[i].append(int(round(h2)))
                                br_list[i].append(int(round(b2)))
                                if len(hr_list[i]) > 5:
                                    hr_list[i].pop(0)
                                if len(br_list[i]) > 5:
                                    br_list[i].pop(0)
                    for i in range(last_num, num):
                        hr_list.append([80, 80, 80, 80, 80])
                        br_list.append([80, 80, 80, 80, 80])
                        d = (index[i] + 50) / 156 + 0.2
                        h1, b1, k_val1, h2, b2, k_val2 = uwb.vital_signs(d - 0.1, d + 0.1)
                        if k_val1 or k_val2:
                            if k_val1 <= k_val2:
                                smaller_k = k_val1
                                hr_list[i].append(int(round(h1)))
                                br_list[i].append(int(round(b1)))
                                if len(hr_list[i]) > 5:
                                    hr_list[i].pop(0)
                                if len(br_list[i]) > 5:
                                    br_list[i].pop(0)
                            else:
                                smaller_k = k_val2
                                phase_flag = 1
                                hr_list[i].append(int(round(h2)))
                                br_list[i].append(int(round(b2)))
                                if len(hr_list[i]) > 5:
                                    hr_list[i].pop(0)
                                if len(br_list[i]) > 5:
                                    br_list[i].pop(0)

                for i in range(num):
                    d = (index[i] + 50) / 156 + 0.2
                    biggest = np.argmax(iou[:, 3])
                    iou_sorted[i] = iou[biggest]
                    iou = np.delete(iou, biggest, 0)
                    x, y, wi, he = iou_sorted[i, 0], iou_sorted[i, 1], iou_sorted[i, 2], iou_sorted[i, 3]
                    mul = 3
                    x_t, y_t, wi_t, he_t = (x - 0.5) * mul + 0.5, (y - 0.5) * mul + 0.5, wi * mul, he * mul
                    
                    x1, x2 = setRoi(x_t - wi_t/2, x_t + wi_t/2, thermal_width)
                    y1, y2 = setRoi(y_t - he_t/2, y_t + he_t/2, thermal_height)
                    # print(x1, x2, y1, y2)
                    t = np.max(t_frame[y1:y2, x1:x2])
                    cv2.rectangle(thermal_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    temp.append(t)


                    # print((int(iou_sorted[i, 0] * cam_width), int(iou_sorted[i, 1] * cam_height)))
                    
                    # cv2.putText(frame, str(h), (int(x * cam_width), int(y * cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    # cv2.putText(frame, str(b), (int(x * cam_width), int(y * cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    # cv2.putText(frame, str(t), (int(x * cam_width), int(y * cam_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    frame = cv2ImgAddText(frame, int((x - wi / 2) * cam_width), int((y - he / 2) * cam_height), np.mean(hr_list[i]), np.mean(br_list[i]), t, d)
            last_num = num

                    
            frame_resized = cv2.resize(frame, (1280, 720))
            # print(frame_resized.shape)
            cv2.imshow('result', frame_resized)
            cv2.imshow('thermal', thermal_image)
            if cv2.waitKey(33) & 0xFF == 27:
                end = True
                uwb.reset()
                cv2.destroyAllWindows()
                break

def camera(opt):
    global hd
    hd = human_detector(opt)

def setRoi(x, y, range):
    x = x * range
    y = y * range
    x = 0 if x < 0 else int(x)
    x = range - 1 if x > range - 1 else int(x)
    y = 0 if y < 0 else int(y)
    y = range - 1 if y > range - 1 else int(y)
    if x == y and x == 0:
        y = y + 1
    if x == y and y == range - 1:
        x = x - 1
    return x, y


def cv2ImgAddText(img, left, top, hr, br, t, d, textColor=(255, 0, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simhei"
        ".ttf", textSize, encoding="utf-8")
    draw.text((left + 10, top+10), str(hr), textColor, font=fontStyle)
    draw.text((left + 10, top+40), str(br), textColor, font=fontStyle)
    draw.text((left + 10, top+70), str(t)+'℃', textColor, font=fontStyle)
    #  **********************************************************************************************
    if resultoxi['1']['oximeter'] == 0:
        resultoxi['1']['oximeter'] = 80
    accuracy_hr = 1 - abs((int(resultoxi['1']['oximeter']) - int(hr))) / int(resultoxi['1']['oximeter'])
    with open("20220531indoor-wyl-with_range.txt", 'a') as f:
        f.write(str(hr))
        f.write(' ' + str(resultoxi['1']['oximeter']))
        f.write(' ' + str(round(100*accuracy_hr, 1)) + '%' + ' b:' + str(br) +
                ' t:' + str(t) + ' k:' + str(smaller_k) + ' PhaseOrNot:' + str(phase_flag) + '\n')
        f.close()
    # **********************************************************************************************
    # draw.text((left, top+100), str(d), textColor, font=fontStyle)
    # draw.text((left, top+50), '血氧仪：'+str(result[target]['oximeter']), textColor, font=fontStyle)


    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

'''
血氧仪的连接与数据传输
Input:
    port_oximeter: 连接血氧仪的端口
Output:
    oximeter_result：血氧仪测得的目标心率
'''
def oximeter(port_oximeter):
    global resultoxi
    tcpCliSock3= socket(AF_INET,SOCK_STREAM)
    tcpCliSock3.setblocking(False)
    tcpCliSock3.bind(('', port_oximeter))
    tcpCliSock3.listen()
    print('oximeter is ready.')
    inputs = [tcpCliSock3, ]
    while 1:
        time.sleep(0.005)
        r_list, w_list, e_list = select.select(inputs, [], [],0.005)
        for event in r_list:
            if event == tcpCliSock3:
                new_sock, addr = event.accept()
                inputs=[tcpCliSock3,new_sock,]
            else:
                data = event.recv(1024)
                # logger.info(data)
                if data!=b'' and data!=b'socket connected':
                    # logger.info(data)
                    oximeter_result=data.split()
                    try:
                        oximeter_tag_num=bytes.decode(oximeter_result[-1])[-1]
                        # if oximeter_list[oximeter_tag_num - 1] in result.keys():
                        resultoxi[oximeter_tag_num]['oximeter'] = int(oximeter_result[-2])
                    except:
                        print('oximeter id wrong')

def thermal():
    global thermal_frame, thermal_image, end
    while not end:
        temp = np.loadtxt(".\\thermal_camera_reading\\data.txt")
        # print(temp.shape)
        if(temp.shape[0] == 110592):
            thermal_frame = np.reshape(temp, (288, 384))
            thermal_image = thermal_frame - np.min(thermal_frame)
            thermal_image = thermal_image / np.max(thermal_image)
            # print(thermal_frame[1, 0])
            # print(thermal_frame)
        sleep(1)

if __name__ == "__main__":
    end = False
    thermal_frame = np.zeros((288, 384))
    thermal_image = np.zeros((288, 384))
    # 清除之前积攒的debug文件
    files = os.listdir()
    for file in files:
        if not os.path.isdir(file):
            if file[0:5] == 'debug':
                os.remove(file)

    os.system('start GuideInfra.exe')

    opt = parse_opt()
    # cam = threading.Thread(name="yolov5", target = camera, args=(opt,))
    # cam.setDaemon(True)
    # cam.start()
    hd = human_detector(opt)
    port_oximeter = 10001
    resultoxi = {
        '1': {'oximeter': 80,
              },
        # '2': {'oximeter': 80,
        #       }
    }
    args = {
        'set_enable':1,
        'set_iterations':64,
        'set_pulses_per_step':38,
        'set_dac_step':1,
        'set_dac_min':949,
        'set_dac_max':1100,
        'set_tx_power':2,
        'set_downconversion':0,
        'set_frame_area_offset':0.18,
        'set_frame_area':[0.2, 1.7],
        'set_tx_center_frequency':3,
        'set_prf_div':16,
        'set_fps':40}
    while 1:
        if hd.work:
            uwb = uwb_radar_with_phase('COM3', args)
            break
        sleep(1)
    smaller_k = 20
    phase_flag = 0
    t = threading.Thread(name="thermal", target=thermal)
    t.start()
    t2 = threading.Thread(name="oximeter", target=oximeter, args=(port_oximeter,))  # 线程2：脉搏血氧仪的连接与数据传输
    t2.start()

    draw()
