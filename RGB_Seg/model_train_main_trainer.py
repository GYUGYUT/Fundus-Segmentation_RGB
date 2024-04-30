import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.functional as F
from pickle import TRUE
from dataloader_dir2 import *
from getmodel  import *
from tqdm import tqdm
from paser_args import *
from pytorchtools import EarlyStopping
from train_test_module import * 
import wandb
from torch.optim.lr_scheduler import StepLR
import segmentation_models_pytorch as smp
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 시드 설정
seed = 42
set_seed(seed)
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"]= "1,2"  # Set the GPU 0 to use

def run():
    
    DEVICE = 'cuda'
    model = getModel(args.arch, cfg["num_class"])
    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="fundus_hyperspectral_segementation",
        entity = "alswo740012"
    )
    wandb.run.name = str("RGB_") + str(args.arch) + "_" + str(cfg["lr"]) + "_" + str(cfg["batch_size"]) + "_" + str(cfg["imgsize"]) + "_" + str(cfg["num_class"]) 
    wandb.run.save()
    config = {"lr":cfg["lr"],"batch_size":cfg["batch_size"],"architecture" : model}
    wandb.init(config = config)
    wandb.watch(model,loss,log="all",log_freq=10)

    train_data_path = r"/home/msl2024a/resize_idrid2/hyperspectral/train"
    test_data_path = r"/home/msl2024a/resize_idrid2/hyperspectral/test"

    Label_train_data_path = r"/home/msl2024a/resize_idrid2/gt/train"
    Label_test_data_path = r"/home/msl2024a/resize_idrid2/gt/test"

    train_data,val_data,test_data = get_loder_main(train_data_path,test_data_path,
                                                   Label_train_data_path,Label_test_data_path,
                                                   cfg["imgsize"],cfg["batch_size"],cfg["shuffle"],cfg["numworks"])
    train_epoch = smp.utils.train.TrainEpoch(
                                            model, 
                                            loss=loss, 
                                            metrics=metrics, 
                                            optimizer=optimizer,
                                            device=DEVICE,
                                            verbose=True,
                                            )

    valid_epoch = smp.utils.train.ValidEpoch(
                                            model, 
                                            loss=loss, 
                                            metrics=metrics, 
                                            device=DEVICE,
                                            verbose=True,
                                        )
    valid_epoch = smp.utils.train.ValidEpoch(
                                            model, 
                                            loss=loss, 
                                            metrics=metrics, 
                                            device=DEVICE,
                                            verbose=True,
                                        )
    
    max_score = 0
    

    for i in range(0, 40):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_data)
        valid_logs = valid_epoch.run(val_data)
        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, './best_model.pth')
            print('Model saved!')
            
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')
    # best_model = torch.load('./best_model.pth')
    # test_epoch = smp.utils.train.ValidEpoch(
    #     model=best_model,
    #     loss=loss,
    #     metrics=metrics,
    #     device=DEVICE,
    # )
    # logs = test_epoch.run(test_data)
    #     test_dataset_vis = Dataset(
    #     x_test_dir, y_test_dir, 
    #     classes=CLASSES,
    # )
    # for i in range(5):
    #     n = np.random.choice(len(test_data))
        
    #     image_vis = test_dataset_vis[n][0].astype('uint8')
    #     image, gt_mask = test_dataset[n]
        
    #     gt_mask = gt_mask.squeeze()
        
    #     x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    #     pr_mask = best_model.predict(x_tensor)
    #     pr_mask = (pr_mask.squeeze().cpu().numpy().round())
            
#hyper param

 
# script = ['densenet121','resnet50','alexnet','resnext50','mobilenet_v2','densenet169','resnet101','vgg19','resnext101'] # 'vgg16'
script = ['UNet'] # 'vgg16','resnext50','densenet121',
batch_sizes = [8]
for i in script:
    for select_batch in batch_sizes:
        #DEVICE
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = None
        cfg = { model : i,
                    "epoch" : 100,
                    "lr" : 4e-4,
                    "batch_size": select_batch,
                    "shuffle" : TRUE,
                    "imgsize" : [512,512],
                    "num_class" : 5,
                    "numworks" : 16,
                    "patience" : 20,
                    "class_name" : ["MA","HM","HE","SE","OD"]}
        args = parse_args(cfg[model],cfg["class_name"])
        a = run()
        del a