import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,f"{BASE_PATH}/deep_high_resolution_net")

import math
import argparse
import csv
import shutil

from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np
import time
import json
from matplotlib import pyplot as plt
import logging


# import common


class DNN:
    """
    static한 걸 여기에 담는다 

    Parameters
    ----------
    threshold : float
        threshold = 0.65
    CLASSES : dictionary
        0: 'STAND'
        1: 'FALLDOWN'

    Returns
    -------
    
    None

    """
    def __init__(self, logger):
        self.logger = logger
        self.threshold = 0.65
        self.CLASSES = {
            0: 'STAND',
            1: 'FALLDOWN',
        }

    # 안 건드리는거!   
    def load(self,weights):
        self.weights = weights
        self.Engine = common.Engine()      
        self.Engine.make_context(self.weights)
        print(f'Engine shape : {self.Engine.input_shape}')
        self.batchsize = int(self.Engine.input_shape[0])
        self.input_c = int(self.Engine.input_shape[1])
        self.input_h = int(self.Engine.input_shape[2])
        self.input_w = int(self.Engine.input_shape[3])


    # 예외적인 추가 함수 

    '''
    input : routing data
                        : { 'bbox' : 
                            'data' : 
                            'framedata' : 'setp_result' : 
                            'available' : }
    otuput : parse_input_result [ [data], [data], ... ]
    '''
    def parse_input(self,input_data_batch):
        
        parse_input_result = []
        for input_data in input_data_batch:

            frame = input_data['framedata']['frame']
            bbox = input_data['bbox']
            # cropped_img = common.getCropByFrame(frame,bbox)
            parse_input_result.append(cropped_img)

        return parse_input_result

    '''
    input : [[cropped_img], [cropped_img],...]
    output : 
    '''
    def preprocess(self, parse_input_result):

        batch_resized_tensor = torch.zeros([len(parse_input_result), self.input_c, self.input_h, self.input_w], dtype=torch.float, device=torch.device("cuda:0"))

        for idx, img_tensor in enumerate(parse_input_result) :
                        
            w = img_tensor.shape[2]
            h = img_tensor.shape[1]
            
            #288,384
            t_w = 278
            t_h = 374

            r = min(278/w, 374/h)

            rh, rw = int(r*h), int(r*w)

            resized_img_tensor = transforms.functional.resize(img_tensor, size=(rh, rw))
            
            left_pad = int((288 - rw) / 2)
            right_pad = 288 - (rw + left_pad)
            left_end = rw + left_pad

            top_pad = int((384 - rh) / 2)
            bottom_pad = 384 - (rh + top_pad)
            top_end = rh + top_pad

            batch_resized_tensor[idx, : , top_pad:top_end, left_pad:left_end] = resized_img_tensor

            
        batch_to_tensor = batch_resized_tensor.div(255) # 빈 값
        
        preprocess_result = transforms.functional.normalize(batch_to_tensor,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        return preprocess_result


    '''
    input : crop image
    output : HRnet + DNN model inferenc result 
    '''
    def inference(self,preprocess_result) :
        result = self.Engine.do_inference_v2(preprocess_result)
        inference_result = result[0]
        return inference_result

    
    '''
    input :
    output : 
    '''    
    def postprocess(self,inference_result):
        postprocess_result = []
        inference_result = inference_result.detach().cpu()
        pred_result = inference_result.max(1, keepdim=True)[1]
        for idx, result in enumerate(pred_result):
            pose_class = int(result)
            pose_result = self.CLASSES[pose_class]
            dnn_score = float(inference_result[idx][pose_class])

            postprocess_result.append([pose_class, pose_result, dnn_score])
        return postprocess_result

    '''
    input :
    output : 
    '''
    def parse_output(self,input_data_batch,output_batch):
        print(f'input_data_batch shape : {len(input_data_batch)}')
        print(f'output_batch : {len(output_batch)}')

        res = []
        for idx_i, data in enumerate(input_data_batch):
            framedata = data['framedata']
            scenario = data['scenario']
            bbox = data['bbox']
            
            if output_batch[idx_i] == None:
                input_data = dict()
                input_data["framedata"] = framedata
                input_data["bbox"] = None
                input_data["scenario"] = scenario
                input_data["data"] = None
                input_data["available"] = False
                res.append(input_data)
                continue

            outputs = output_batch[idx_i]
            print('outputs', outputs)
            
            # 하나의 크롭이미지에서 겹쳐있는 사람 대비용으로
            for output in outputs:
            
                if isinstance(output, type(None)):
                    input_data = dict()
                    input_data["framedata"] = framedata
                    input_data["bbox"] = None
                    input_data["scenario"] = scenario
                    input_data["data"] = None
                    input_data["available"] = False
                    res.append(input_data)
                    continue

                input_data = dict()
                input_data["framedata"] = framedata
                input_data["bbox"] = bbox
                input_data["scenario"] = scenario
                input_data["data"] = output
                input_data["available"] = True
                res.append(input_data)
            
        return res

    def run_inference(self, input_data_batch):
        parse_input_result = self.parse_input(input_data_batch)
        preprocess_result = self.preprocess(parse_input_result)
        inference_result = self.inference(preprocess_result)
        postprocess_result = self.postprocess(inference_result)
        output_data_batch = self.parse_output(input_data_batch, postprocess_result)
        return output_data_batch
    
    

def module_load(logger):
    cls = DNN(logger)
    return cls  

if __name__ == '__main__':
    import logging
    logger = logging.Logger('inference')

    img_ori = cv2.imread("/vidigo/nfs/falldown_img.jpg")
    img = torch.tensor(img_ori).cuda()
    img = img.permute(2,0,1)

    ## 더미데이터 생성
    input_data = dict()
    input_data["framedata"] = {"frame":img}
    input_data["bbox"] = [0,0,img.shape[2],img.shape[1]]
    input_data["scenario"] = "s"
    input_data["data"] = None
    
    base_path = os.path.dirname(os.path.realpath(__file__))
    dnn_weights = os.path.join(base_path, '/vidigo/model_manager/engines/falldown_hrnet_dnn/falldown_hrnet_dnn_fp16_032.trt')
    print(os.path.exists(dnn_weights))

    dnn = module_load(logger)
    print('load_start')
    dnn.load(dnn_weights)

    # 가상데이터를 똑같은 걸 배치사이즈 8로 해서 넣은거!
    input_data_batch = [input_data for i in range(8)]

    ## 추론 시작
    output = dnn.run_inference(input_data_batch)
    print(f"output : {output},{len(output)}")