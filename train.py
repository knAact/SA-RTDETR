import warnings, os

warnings.filterwarnings('ignore')
from ultralytics import RTDETR



if __name__ == '__main__':
    model = RTDETR('./SA-RTDETR.yaml')
    # model.load('/root/workspace/RTDETR-main/weights/rtdetr-r18.pt') # loading pretrain weights
    model.train(data='./dataset/VisDrone_YOLO/VisDrone_Dataset.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=4,
                workers=4,
                # device='0,1',
                # resume='', # last.pt path
                project='runs/train',
                name='exp',
                )