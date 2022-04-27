# created by Xikang Jiang
from pymoduleconnector import ModuleConnector
import logging
import numpy as np
import sys
from VMDHRBR import VMDHRBR
import pca_filter
import threading
from time import sleep
import time
import os
import scipy.io as sio


class uwb_radar:
    leiji = None
    leiji_data_storage = None
    pure_data_storage = None
    pure_data = None
    xep = None
    can_read = False
    read_condition = None
    end = False
    list_length = 0
    range = [0, 0]
    now_read = None

    def __init__(self, com, kwargs):
        self.init_radar(com, kwargs)
        print('End init')

    def init_radar(self, com, kwargs):
        self.range = kwargs['set_frame_area']
        self.mc = ModuleConnector(com, log_level=0)
        self.xep = self.mc.get_xep()

        # inti x4driver
        self.xep.x4driver_init()
        # print('initing')

        # Set enable pin
        self.xep.x4driver_set_enable(kwargs['set_enable'])

        # Set iterations
        self.xep.x4driver_set_iterations(kwargs['set_iterations'])
        # Set pulses per step
        self.xep.x4driver_set_pulses_per_step(kwargs['set_pulses_per_step'])
        # Set dac step
        self.xep.x4driver_set_dac_step(kwargs['set_dac_step'])
        # Set dac min
        self.xep.x4driver_set_dac_min(kwargs['set_dac_min'])
        # Set dac max
        self.xep.x4driver_set_dac_max(kwargs['set_dac_max'])
        # Set TX power
        self.xep.x4driver_set_tx_power(kwargs['set_tx_power'])

        # Enable downconversion
        self.xep.x4driver_set_downconversion(kwargs['set_downconversion'])

        # Set frame area offset
        self.xep.x4driver_set_frame_area_offset(kwargs['set_frame_area_offset'])
        offset = self.xep.x4driver_get_frame_area_offset()

        # Set frame area
        self.xep.x4driver_set_frame_area(self.range[0], self.range[1])
        frame_area = self.xep.x4driver_get_frame_area()

        # Set TX center freq
        self.xep.x4driver_set_tx_center_frequency(kwargs['set_tx_center_frequency'])

        # Set PRFdiv
        self.xep.x4driver_set_prf_div(kwargs['set_prf_div'])
        prf_div = self.xep.x4driver_get_prf_div()

        # Start streaming
        self.xep.x4driver_set_fps(kwargs['set_fps'])
        fps = self.xep.x4driver_get_fps()

        print("wait radar")
        self.read_condition = threading.Condition()

        self.list_length = len(self.read_frame())

        self.read_frame_circulation = threading.Thread(
            name='read_frame_circulation', target=self.read_circulation, args=())
        # read_frame_circulation.setDaemon(True)
        self.preprocess = threading.Thread(
            name='preprocess', target=self.preprocessing, args=())
        # preprocess.setDaemon(True)
        self.data_storage= threading.Thread(
            name='data_storage', target=self.data_storage_raw, args=())
        self.read_frame_circulation.setDaemon(True)
        self.preprocess.setDaemon(True)
        self.data_storage.setDaemon(True)
        self.read_frame_circulation.start()
        self.preprocess.start()
        self.data_storage.start()

    def preprocessing(self):
        ycl = 1
        M = 20  # 输出结果去除前M行
        L = 50  # 输出结果去除前L列，同时后50列也会被去掉
        # global leiji, pure_data
        print('ready')
        while not self.end:
            sleep(0.1)
            if True:
            # with self.read_condition:
                # print(self.leiji.shape[0])
                if self.leiji is not None and self.leiji.shape[0] > 240:
                    if ycl == 1:
                        print('init')
                    self.leiji = self.leiji[-220:, :]
                    rawdata = self.leiji[-220:, :-1].copy()
                    self.pure_data = pca_filter.p_f(rawdata, M, L)
                    ycl += 1
                    self.can_read = True
                # self.read_condition.notify()

    def read_circulation(self):
        # global leiji, pure_data
        while not self.end:
            if True:
            # with self.read_condition:
                # print(hex(self.xep.ping()))
                start = time.time()
                save = self.get_data()
                if self.leiji is None:
                    self.leiji = save.copy()
                    # print(leiji)
                else:
                    self.leiji = np.vstack((self.leiji, save))
                self.now_read = save.copy()
                # print('Read frame')
                # self.read_condition.notify()
                # print(time.time() - start)
            sleep(0.001)

    def read_frame(self):
        """Gets frame data from module"""
        d = self.xep.read_message_data_float()
        frame = np.array(d.data)
        # Convert the resulting frame to a complex array if downconversion is enabled
        return frame

    def get_data(self):
        save1 = np.ones((1, self.list_length))
        # print(len(self.read_frame()))

        if not self.end:
            
            frame2 = self.read_frame()

            save1[0, :] = frame2
            # ll2=time.clock()-start
            # ll2 = time.asctime(time.localtime(time.time()))
            #
            # ll21 = ll2[11:13]
            # ll22 = ll2[14:16]
            # ll23 = ll2[17:19]
            #
            # lll = int(ll21) * 10000 + int(ll22) * 100 + int(ll23)
            # save[jj, -1] = lll
            # print(leiji.shape[0],lll)
        return save1

    def data_storage_raw(self):
        num = 1
        while not self.end:
            save2 = self.get_data()
            if self.leiji_data_storage is None:
                self.leiji_data_storage = save2.copy()
            else:
                self.leiji_data_storage = np.vstack((self.leiji_data_storage,save2))
            if self.leiji_data_storage.shape[0] >= 20 * 60 * 1:
                sio.savemat('D:/AllAboutPostGraduateWork/项目/301医院/二阶段/MCI_monitor/data_storage/20220427indoorwyl{num}.mat'.format(num=num), {'data': self.leiji_data_storage})
                self.leiji_data_storage = None
                num += 1

    def data_storage_pure(self):
        pure_num = 1
        while not self.end:
            save3 = self.pure_data
            self.pure_data_storage = save3.copy()
            if self.pure_data_storage.shape[0] >= 20 * 60 * 1:
                sio.savemat("D:/AllAboutPostGraduateWork/项目/301医院/二阶段/MCI_monitor/pure_data_storage/20220423testing.mat", {'data{num}'.format(num=pure_num): self.pure_data_storage})
                self.pure_data_storage = None
                pure_num += 1

    def max_energy(self, puredata):
        data = puredata
        energyOfLocation = [0] * data.shape[1]
        for i in range(data.shape[1]):
            energyOfLocation[i] = sum(data[:, i] * data[:, i])
        sig = puredata[:, np.argmax(energyOfLocation)]
        return sig

    # # 每一行的最大值
    # def max_energy(self, puredata):
    #     data = puredata.copy()
    #     sig = np.zeros(data.shape[0])
    #     for i in range(data.shape[0]):
    #         sig[i] = max(data[i])
    #     return sig.tolist()

    def vital_signs(self, range_low, range_high):
        # if range_low <= self.range[0] or range_high >= self.range[1]:
        #     return 0, 0
        beginindex = round((range_low - self.range[0])*156 - 50)
        endindex = round((range_high - self.range[0])*156 - 50)
        beginindex = 0 if beginindex<0 else beginindex
        endindex = 0 if endindex<0 else endindex

        if self.can_read and self.pure_data is not None:
            beginindex = self.pure_data.shape[1] if beginindex>self.pure_data.shape[1] else beginindex
            endindex = self.pure_data.shape[1] if endindex>self.pure_data.shape[1] else endindex
            if beginindex == endindex:
                return 0, 0
            # print(beginindex)
            # print(endindex)
            PureData3 = self.pure_data[:, int(beginindex):int(endindex)].copy()
            sig1 = self.max_energy(PureData3)
            out = VMDHRBR(sig1)
            hr = int(round(out[0]))
            br = int(round(out[1]))
            return hr, br
        else:
            return 0, 0

    def reset(self):
        # self.count = self.count + 1
        # print('destruction', self.count)
        self.end = True
        # self.read_frame_circulation.join()
        # self.preprocess.join()
        if self.preprocess.is_alive():
            self.preprocess.join()
        if self.read_frame_circulation.is_alive():
            self.read_frame_circulation.join()
        if self.data_storage.is_alive():
            self.data_storage.join()

        # while self.read_frame_circulation.is_alive() or self.preprocess.is_alive():
        #     sleep(0.1)
        self.xep.module_reset()
        print("radar reset!")
        return 1

    def __del__(self):
        print('destruction')
        if self.end != True:
            self.end = True
            
            if hasattr(self, 'read_frame_circulation'):
                if self.read_frame_circulation.is_alive():
                    self.read_frame_circulation.join()
            if hasattr(self, 'preprocess'):
                if self.preprocess.is_alive():
                    self.preprocess.join()
            if hasattr(self, 'data_storage'):
                if self.data_storage.is_alive():
                    self.data_storage.join()
            
            self.xep.module_reset()


if __name__ == '__main__':
    # 清除之前积攒的debug文件
    files = os.listdir()
    for file in files:
        if not os.path.isdir(file):
            if file[0:5] == 'debug':
                os.remove(file)

    # os.system('start GuideInfra.exe')

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
        'set_frame_area':[0.2, 4],
        'set_tx_center_frequency':3,
        'set_prf_div':16,
        'set_fps':20}
    uwb = uwb_radar('COM8', args)
    # print('Start reading')
    i = 0
    while i < 600:
        # if uwb.PureData.shape[0] > 1 and uwb.can_read:
        #     if result[tag]['state'] and (result[tag]['endindex'] != result[tag]['beginindex']) :
        #         PureData3 = uwb.PureData[:, result[tag]['beginindex']:result[tag]['endindex']].copy()
        #         tmp_vs = uwb.vital_signs(PureData3)
        #         result[tag]['heartbeat'] = tmp_vs[0] if tmp_vs[0] != -1 else result[tag]['heartbeat']
        #         result[tag]['breath'] = tmp_vs[1] if tmp_vs[1] != -1 else result[tag]['breath']
        #         # result[tag]['heartbeat']=tmp_vs[0] if tmp_vs[0] != -1 else None
        #         # result[tag]['breath']=tmp_vs[1] if tmp_vs[1] != -1 else None
        # print(uwb.vital_signs(110, 130)
        hr = uwb.vital_signs(0, 1)[0]
        if hr != -1:
            print(hr)
        sleep(0.1)
        i += 1
        # print(i + 1)
        # print('ready')
        pass
    uwb.reset()
