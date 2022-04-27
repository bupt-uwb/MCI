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

from uwb_radar import uwb_radar
from human_detection import human_detector
a=100
b=1
 

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
    global hd, uwb, thermal_frame, thermal_image, end
    # global hd

    while 1:
        # sleep(0.01)
        # print(hd.work)
        if hd.work: 
            # frame = frame[:,:,::-1]
            isman_camera = hd.isman_camera
            num = hd.num
            # print('num:')
            # print(num)                                                             
            iou = np.array(hd.iou)
            iou_sorted = iou.copy()
            frame = hd.frame_output
            t_frame = thermal_frame.copy()
            cam_width = frame.shape[1]
            cam_height = frame.shape[0]
            thermal_width = 384
            thermal_height = 288
            # print('iou')
            # print(iou.shape)

            if isman_camera and uwb.can_read:
                amplitude, index, energy_ratio = tools.sp(uwb.pure_data[-1, :], num)
                # print(index)

                hr = []
                br = []
                temp = []
                for i in range(num):
                    d = (index[i] + 50) / 156 + 0.2
                    h, b = uwb.vital_signs(d - 0.1, d + 0.1)
                    if h == -1:
                        h = 0
                    if b == -1:
                        b = 0
                    hr.append(h)
                    br.append(b)

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
                    frame = cv2ImgAddText(frame, int((x - wi / 2) * cam_width), int((y - he / 2) * cam_height), h, b, t, d)
                    
                    # cv2.rectangle(frame, )

                # print('hr')
                # print(hr)
                # print('br')
                # print(br)
                    
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
    # 绘制文本
    # if result[target]['state'] and result[target]['heartbeat']>0:
        # draw.text((left, top+10), '目标：'+target, textColor, font=fontStyle)
        # draw.text((left, top+30), '距离：'+str(d), font=fontStyle)

    # draw.text((left, top+10), '心率'+str(hr), textColor, font=fontStyle)
    # draw.text((left, top+40), '呼吸率'+str(br), textColor, font=fontStyle)
    # draw.text((left, top+70), '温度'+str(t)+'℃', textColor, font=fontStyle)
    draw.text((left + 10, top+10), str(hr), textColor, font=fontStyle)
    draw.text((left + 10, top+40), str(br), textColor, font=fontStyle)
    draw.text((left + 10, top+70), str(t)+'℃', textColor, font=fontStyle)

    # draw.text((left, top+100), str(d), textColor, font=fontStyle)
    # draw.text((left, top+50), '血氧仪：'+str(result[target]['oximeter']), textColor, font=fontStyle)

    # 转换回OpenCV格式
    # img.show()
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

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

    args = {
        'set_enable':1,
        'set_iterations':64,
        'set_pulses_per_step':5,
        'set_dac_step':1,
        'set_dac_min':949,
        'set_dac_max':1100,
        'set_tx_power':2,
        'set_downconversion':0,
        'set_frame_area_offset':0.18,
        'set_frame_area':[0.2, 5],
        'set_tx_center_frequency':3,
        'set_prf_div':16,
        'set_fps':20}
    while 1:
        if hd.work:
            uwb = uwb_radar('COM3', args)
            break
        sleep(1)

    t = threading.Thread(name="thermal", target=thermal)
    t.start()

    draw()
