import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from tqdm import tqdm
from confusion_matrix import *
from report_save import *
from cal_accuracy import *
from roc_auc import *
from sklearn.metrics import precision_recall_fscore_support
import segmentation_models_pytorch as smp
import torch.nn.functional as F

def cal_matrix(output, target, mode='binary', threshold=0.5):
    tp, fp, fn, tn = smp.metrics.get_stats(output.round().long(), target.round().long(), mode=mode, threshold=threshold)

    iou_score = smp.metrics.iou_score(tp, fp, fn, tn)
    recall = smp.metrics.recall(tp, fp, fn, tn)

    precision = smp.metrics.precision(tp, fp, fn, tn)
    sensitivity = smp.metrics.sensitivity(tp, fp, fn, tn)
    specificity = smp.metrics.specificity(tp, fp, fn, tn)

    f1_score = smp.metrics.f1_score(tp, fp, fn, tn)
    accuracy = smp.metrics.accuracy(tp, fp, fn, tn)

    return [iou_score.cpu(),recall.cpu(),precision.cpu(),sensitivity.cpu(),specificity.cpu(),f1_score.cpu(),accuracy.cpu() ]
    

def train(device,args,model,train_data,val_data,loss_fn, optimizer, epoch,wandb,early_stopping):
    model.train()
    loss = 0.0
    total_loss = []
    for batch_id, (X, y) in enumerate(tqdm(train_data,"Train_epoch : %d, lr : %f, progress "% (epoch, optimizer.param_groups[0]['lr']))):
        X, y = X.to(device), y.to(device)

        pred = model(X)

        # pred = create_feature_map(pred)
        loss = loss_fn(pred, y)

        total_loss.append(loss.cpu().item())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
    wandb.log({"train_loss ": np.mean(total_loss)},step=epoch)    
    print("epoch {} batch id {} loss {:.6f}".format(epoch, batch_id+1, np.mean(total_loss)))


    val_loss = 0.0
    model.eval()
    val_total_loss = []

    iou_score = []
    iou_score1 = []
    iou_score2 = []
    iou_score3 = []
    iou_score4 = []
    iou_score5 = []

    recall_list = []
    recall_list1 = []
    recall_list2 = []
    recall_list3 = []
    recall_list4 = []
    recall_list5 = []

    precision_list = []
    precision_list1 = []
    precision_list2 = []
    precision_list3 = []
    precision_list4 = []
    precision_list5 = []

    sensitivity_list = []
    sensitivity_list1 = []
    sensitivity_list2 = []
    sensitivity_list3 = []
    sensitivity_list4 = []
    sensitivity_list5 = []

    specificity_list = []
    specificity_list1 = []
    specificity_list2 = []
    specificity_list3 = []
    specificity_list4 = []
    specificity_list5 = []

    f1_list = [] 
    f1_list1 = [] 
    f1_list2 = [] 
    f1_list3 = [] 
    f1_list4 = [] 
    f1_list5 = [] 
    
    acc_list = []
    acc_list1 = []
    acc_list2 = []
    acc_list3 = []
    acc_list4 = []
    acc_list5 = []


    

    for batch_id, (X, y) in enumerate(tqdm(val_data," %d val!!!!"% epoch)):
        example_image = []
        X, y = X.to(device), y.to(device)

        with torch.no_grad():
            model.training = False
            # Compute prediction error
            pred = model(X)
            
            val_loss = loss_fn(pred, y)
            pred = torch.where(pred > 0.5, torch.tensor(1.0), torch.tensor(0.0))
            # pred=create_feature_map(pred)
            val_total_loss.append(val_loss.item())

        matrix = cal_matrix(pred, y)
        iou_score.append(matrix[0])
        recall_list.append(matrix[1])
        precision_list.append(matrix[2])
        sensitivity_list.append(matrix[3])
        specificity_list.append(matrix[4])
        f1_list.append(matrix[5])
        acc_list.append(matrix[6])
        for image_idx in range(len(pred)):
            
            for ch_idx in range(len(pred[image_idx])):

                if ch_idx == 0 :
                    matrix = cal_matrix(pred[image_idx][ch_idx], y[image_idx][ch_idx])

                    iou_score1.append(matrix[0])
                    recall_list1.append(matrix[1])
                    precision_list1.append(matrix[2])
                    sensitivity_list1.append(matrix[3])
                    specificity_list1.append(matrix[4])
                    f1_list1.append(matrix[5])
                    acc_list1.append(matrix[6])
                    
                elif ch_idx == 1 :
                    matrix = cal_matrix(pred[image_idx][ch_idx], y[image_idx][ch_idx])

                    iou_score2.append(matrix[0])
                    recall_list2.append(matrix[1])
                    precision_list2.append(matrix[2])
                    sensitivity_list2.append(matrix[3])
                    specificity_list2.append(matrix[4])
                    f1_list2.append(matrix[5])
                    acc_list2.append(matrix[6])
                elif ch_idx == 2 :
                    matrix = cal_matrix(pred[image_idx][ch_idx], y[image_idx][ch_idx])

                    iou_score3.append(matrix[0])
                    recall_list3.append(matrix[1])
                    precision_list3.append(matrix[2])
                    sensitivity_list3.append(matrix[3])
                    specificity_list3.append(matrix[4])
                    f1_list3.append(matrix[5])
                    acc_list3.append(matrix[6])
                elif ch_idx == 3 :
                    matrix = cal_matrix(pred[image_idx][ch_idx], y[image_idx][ch_idx])
                    iou_score4.append(matrix[0])
                    recall_list4.append(matrix[1])
                    precision_list4.append(matrix[2])
                    sensitivity_list4.append(matrix[3])
                    specificity_list4.append(matrix[4])
                    f1_list4.append(matrix[5])
                    acc_list4.append(matrix[6])
                elif ch_idx == 4 :
                    matrix = cal_matrix(pred[image_idx][ch_idx], y[image_idx][ch_idx])

                    iou_score5.append(matrix[0])
                    recall_list5.append(matrix[1])
                    precision_list5.append(matrix[2])
                    sensitivity_list5.append(matrix[3])
                    specificity_list5.append(matrix[4])
                    f1_list5.append(matrix[5])
                    acc_list5.append(matrix[6])

    print("-------------->val loss {:.6f}".format(np.mean(val_total_loss)))
    print("val_mean_iou_epoch:", np.mean(iou_score))
    print("val_mean_recall_epoch:", np.mean(recall_list))
    print("val_mean_precision_epoch:", np.mean(precision_list))
    print("val_mean_sensitivity_epoch:", np.mean(sensitivity_list))
    print("val_mean_specificity_epoch:", np.mean(specificity_list))
    print("val_mean_f1_epoch:", np.mean(f1_list))
    print("val_mean_acc_epoch:", np.mean(acc_list))
    print("----"*5)
    print("val_mean_iou_1:", np.mean(iou_score1))
    print("val_mean_recall_1:", np.mean(recall_list1))
    print("val_mean_precision_1:", np.mean(precision_list1))
    print("val_mean_sensitivity_1:", np.mean(sensitivity_list1))
    print("val_mean_specificity_1:", np.mean(specificity_list1))
    print("val_mean_f1_1:", np.mean(f1_list1))
    print("val_mean_acc_1:", np.mean(acc_list1))
    print("----"*5)
    print("val_mean_iou_2", np.mean(iou_score2))
    print("val_mean_recall_2", np.mean(recall_list2))
    print("val_mean_precision_2:", np.mean(precision_list2))
    print("val_mean_sensitivity_2:", np.mean(sensitivity_list2))
    print("val_mean_specificity_2:", np.mean(specificity_list2))
    print("val_mean_f1_2:", np.mean(f1_list2))
    print("val_mean_acc_2:", np.mean(acc_list2))
    print("----"*5)
    print("val_mean_iou_3", np.mean(iou_score3))
    print("val_mean_recall_3", np.mean(recall_list3))
    print("val_mean_precision_3:", np.mean(precision_list3))
    print("val_mean_sensitivity_3:", np.mean(sensitivity_list3))
    print("val_mean_specificity_3:", np.mean(specificity_list3))
    print("val_mean_f1_3:", np.mean(f1_list3))
    print("val_mean_acc_3:", np.mean(acc_list3))
    print("----"*5)
    print("val_mean_iou_4", np.mean(iou_score4))
    print("val_mean_recall_4", np.mean(recall_list4))
    print("val_mean_precision_4:", np.mean(precision_list4))
    print("val_mean_sensitivity_4:", np.mean(sensitivity_list4))
    print("val_mean_specificity_4:", np.mean(specificity_list4))
    print("val_mean_f1_4:", np.mean(f1_list4))
    print("val_mean_acc_4:", np.mean(acc_list4))
    print("----"*5)
    print("val_mean_iou_5", np.mean(iou_score5))
    print("val_mean_recall_5", np.mean(recall_list5))
    print("val_mean_precision_5:", np.mean(precision_list5))
    print("val_mean_sensitivity_5:", np.mean(sensitivity_list5))
    print("val_mean_specificity_5:", np.mean(specificity_list5))
    print("val_mean_f1_5:", np.mean(f1_list5))
    print("val_mean_acc_5:", np.mean(acc_list5))
    print("----"*5)


    example_image.append(wandb.Image(pred[0][0],caption="Pred"))
    example_image.append(wandb.Image(y[0][0],caption="true"))      

    example_image.append(wandb.Image(pred[0][1],caption="Pred2"))
    example_image.append(wandb.Image(y[0][1],caption="true2"))   


    example_image.append(wandb.Image(pred[0][2],caption="Pred3"))
    example_image.append(wandb.Image(y[0][2],caption="true3"))      

    example_image.append(wandb.Image(pred[0][3],caption="Pred4"))
    example_image.append(wandb.Image(y[0][3],caption="true4"))     


    example_image.append(wandb.Image(pred[0][4],caption="Pred5"))
    example_image.append(wandb.Image(y[0][4],caption="true5"))    
    wandb.log({"val_Exampes":example_image,
               "val_loss" : np.mean(val_total_loss),

               "val_mean_iou_epoch" : np.mean(iou_score),
               "val_mean_recall_epoch" : np.mean(recall_list),
               "val_mean_precision_epoch":np.mean(precision_list),
               "val_mean_sensitivity_epoch":np.mean(sensitivity_list),
               "val_mean_specificity_epoch":np.mean(specificity_list),
               "val_mean_f1_epoch":np.mean(f1_list),
               "val_mean_acc_epoch":np.mean(acc_list),

               "val_mean_iou_1" : np.mean(iou_score1),
               "val_mean_recall_1" : np.mean(recall_list1),
               "val_mean_precision_1":np.mean(precision_list1),
               "val_mean_sensitivity_1":np.mean(sensitivity_list1),
               "val_mean_specificity_1":np.mean(specificity_list1),
               "val_mean_f1_1":np.mean(f1_list1),
               "val_mean_acc_1":np.mean(acc_list1),

               "val_mean_iou_2" : np.mean(iou_score2),
               "val_mean_recall_2" : np.mean(recall_list2),
               "val_mean_precision_2":np.mean(precision_list2),
               "val_mean_sensitivity_2":np.mean(sensitivity_list2),
               "val_mean_specificity_2":np.mean(specificity_list2),
               "val_mean_f1_2":np.mean(f1_list2),
               "val_mean_acc_2":np.mean(acc_list2),

               "val_mean_iou_3" : np.mean(iou_score3),
               "val_mean_recall_3" : np.mean(recall_list3),
               "val_mean_precision_3":np.mean(precision_list3),
               "val_mean_sensitivity_3":np.mean(sensitivity_list3),
               "val_mean_specificity_3":np.mean(specificity_list3),
               "val_mean_f1_3":np.mean(f1_list3),
               "val_mean_acc_3":np.mean(acc_list3),

               "val_mean_iou_4" : np.mean(iou_score4),
               "val_mean_recall_4" : np.mean(recall_list4),
               "val_mean_precision_4":np.mean(precision_list4),
               "val_mean_sensitivity_4":np.mean(sensitivity_list4),
               "val_mean_specificity_4":np.mean(specificity_list4),
               "val_mean_f1_4":np.mean(f1_list4),
               "val_mean_acc_4":np.mean(acc_list4),

               "val_mean_iou_5" : np.mean(iou_score5),
               "val_mean_recall_5" : np.mean(recall_list5),
               "val_mean_precision_5":np.mean(precision_list5),
               "val_mean_sensitivity_5":np.mean(sensitivity_list5),
               "val_mean_specificity_5":np.mean(specificity_list5),
               "val_mean_f1_5":np.mean(f1_list5),
               "val_mean_acc_5":np.mean(acc_list5),

                })
    
    early_stopping(np.mean(total_loss), model,args,epoch)
    
    
    if early_stopping.early_stop:
        print("Early stopping")


def test(device,args, model, dataloader,loss_fn,wandb,early_stopping, check_GPU_STATE):

    if check_GPU_STATE == 2:
        print("GPU USE 2EA")
        print("best_model : ",early_stopping.path2)
        model.load_state_dict(torch.load(early_stopping.path2))
    else:
        print("GPU USE 1EA")
        print("best_model : ",early_stopping.path)
        model.load_state_dict(torch.load(early_stopping.path)) 

    test_loss = 0.0
    model.eval()
    test_total_loss = []

    # file_path = os.path.join(args.save_path_report, '{}_report.txt'.format(args.arch))
 
    
    

    iou_score1 = []
    iou_score2 = []
    iou_score3 = []
    iou_score4 = []
    iou_score5 = []

    recall_list1 = []
    recall_list2 = []
    recall_list3 = []
    recall_list4 = []
    recall_list5 = []

    precision_list1 = []
    precision_list2 = []
    precision_list3 = []
    precision_list4 = []
    precision_list5 = []

    sensitivity_list1 = []
    sensitivity_list2 = []
    sensitivity_list3 = []
    sensitivity_list4 = []
    sensitivity_list5 = []

    specificity_list1 = []
    specificity_list2 = []
    specificity_list3 = []
    specificity_list4 = []
    specificity_list5 = []

    f1_list1 = [] 
    f1_list2 = [] 
    f1_list3 = [] 
    f1_list4 = [] 
    f1_list5 = [] 

    acc_list1 = []
    acc_list2 = []
    acc_list3 = []
    acc_list4 = []
    acc_list5 = []
    example_image = []
    for batch_id, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            model.training = False
            # Compute prediction error
            pred = model(X)
            
            test_loss = loss_fn(pred.to(device), y.to(device))
            pred = torch.where(pred > 0.5, torch.tensor(1.0), torch.tensor(0.0))
            # pred=create_feature_map(pred)
            test_total_loss.append(test_loss.item())
   
            
            for image_idx in range(len(pred)):
                for ch_idx in range(len(pred[image_idx])):

                    if ch_idx == 0 :
                        matrix = cal_matrix(pred[image_idx][ch_idx], y[image_idx][ch_idx])

                        iou_score1.append(matrix[0])
                        recall_list1.append(matrix[1])
                        precision_list1.append(matrix[2])
                        sensitivity_list1.append(matrix[3])
                        specificity_list1.append(matrix[4])
                        f1_list1.append(matrix[5])
                        acc_list1.append(matrix[6])
                    elif ch_idx == 1 :
                        matrix = cal_matrix(pred[image_idx][ch_idx], y[image_idx][ch_idx])

                        iou_score2.append(matrix[0])
                        recall_list2.append(matrix[1])
                        precision_list2.append(matrix[2])
                        sensitivity_list2.append(matrix[3])
                        specificity_list2.append(matrix[4])
                        f1_list2.append(matrix[5])
                        acc_list2.append(matrix[6])
                    elif ch_idx == 2 :
                        matrix = cal_matrix(pred[image_idx][ch_idx], y[image_idx][ch_idx])

                        iou_score3.append(matrix[0])
                        recall_list3.append(matrix[1])
                        precision_list3.append(matrix[2])
                        sensitivity_list3.append(matrix[3])
                        specificity_list3.append(matrix[4])
                        f1_list3.append(matrix[5])
                        acc_list3.append(matrix[6])
                    elif ch_idx == 3 :
                        matrix = cal_matrix(pred[image_idx][ch_idx], y[image_idx][ch_idx])

                        iou_score4.append(matrix[0])
                        recall_list4.append(matrix[1])
                        precision_list4.append(matrix[2])
                        sensitivity_list4.append(matrix[3])
                        specificity_list4.append(matrix[4])
                        f1_list4.append(matrix[5])
                        acc_list4.append(matrix[6])
                    elif ch_idx == 4 :
                        matrix = cal_matrix(pred[image_idx][ch_idx], y[image_idx][ch_idx])

                        iou_score5.append(matrix[0])
                        recall_list5.append(matrix[1])
                        precision_list5.append(matrix[2])
                        sensitivity_list5.append(matrix[3])
                        specificity_list5.append(matrix[4])
                        f1_list5.append(matrix[5])
                        acc_list5.append(matrix[6])

                    example_image.append(wandb.Image(pred[image_idx][ch_idx],caption=str(image_idx)+"_"+str(ch_idx)+"_test_Pred"))
                    example_image.append(wandb.Image(y[image_idx][ch_idx],caption=str(image_idx)+"_"+str(ch_idx)+"_test_true"))           

    iou_score = iou_score1 + iou_score2 +iou_score2 +iou_score4 +iou_score5
    recall_list = recall_list1 + recall_list2 + recall_list3 + recall_list4 + recall_list5
    precision_list = precision_list1 + precision_list2 + precision_list3 + precision_list4 + precision_list5
    sensitivity_list = sensitivity_list1 + sensitivity_list2 + sensitivity_list3 + sensitivity_list4 + sensitivity_list5
    specificity_list = specificity_list1 + specificity_list2 + specificity_list3 + specificity_list4 + specificity_list5
    f1_list = f1_list1 + f1_list2 + f1_list3 + f1_list4 + f1_list5
    acc_list = acc_list1 + acc_list2 + acc_list3 + acc_list4 + acc_list5

    print("-------------->test loss {:.6f}".format(np.mean(test_total_loss)))
    print("test_mean_iou_total:", np.mean(iou_score))
    print("test_mean_recall_total:", np.mean(recall_list))
    print("test_mean_precision_total:", np.mean(precision_list))
    print("test_mean_sensitivity_total:", np.mean(sensitivity_list))
    print("test_mean_specificity_total:", np.mean(specificity_list))
    print("test_mean_f1_total:", np.mean(f1_list1))
    print("test_mean_acc_total:", np.mean(acc_list1))
    print("----"*5)
    print("test_mean_iou_1:", np.mean(iou_score1))
    print("test_mean_recall_1:", np.mean(recall_list1))
    print("test_mean_precision_1:", np.mean(precision_list1))
    print("test_mean_sensitivity_1:", np.mean(sensitivity_list1))
    print("test_mean_specificity_1:", np.mean(specificity_list1))
    print("test_mean_f1_1:", np.mean(f1_list1))
    print("test_mean_acc_1:", np.mean(acc_list1))
    print("----"*5)
    print("test_mean_iou_2:", np.mean(iou_score2))
    print("test_mean_recall_2:", np.mean(recall_list2))
    print("test_mean_precision_2:", np.mean(precision_list2))
    print("test_mean_sensitivity_2:", np.mean(sensitivity_list2))
    print("test_mean_specificity_2:", np.mean(specificity_list2))
    print("test_mean_f1_2:", np.mean(f1_list2))
    print("test_mean_acc_2:", np.mean(acc_list2))
    print("----"*5)
    print("test_mean_iou_3:", np.mean(iou_score3))
    print("test_mean_recall_3:", np.mean(recall_list3))
    print("test_mean_precision_3:", np.mean(precision_list3))
    print("test_mean_sensitivity_3:", np.mean(sensitivity_list3))
    print("test_mean_specificity_3:", np.mean(specificity_list3))
    print("test_mean_f1_3:", np.mean(f1_list3))
    print("test_mean_acc_3:", np.mean(acc_list3))
    print("----"*5)
    print("test_mean_iou_4:", np.mean(iou_score4))
    print("test_mean_recall_4:", np.mean(recall_list4))
    print("test_mean_precision_4:", np.mean(precision_list4))
    print("test_mean_sensitivity_4:", np.mean(sensitivity_list4))
    print("test_mean_specificity_4:", np.mean(specificity_list4))
    print("test_mean_f1_4:", np.mean(f1_list4))
    print("test_mean_acc_4:", np.mean(acc_list4))
    print("----"*5)
    print("test_mean_iou_5:", np.mean(iou_score5))
    print("test_mean_recall_5:", np.mean(recall_list5))
    print("test_mean_precision_5:", np.mean(precision_list5))
    print("test_mean_sensitivity_5:", np.mean(sensitivity_list5))
    print("test_mean_specificity_5:", np.mean(specificity_list5))
    print("test_mean_f1_5:", np.mean(f1_list5))
    print("test_mean_acc_5:", np.mean(acc_list5))
    print("----"*5)

  
    wandb.log({"test_Exampes":example_image,
               "test_loss" : np.mean(test_total_loss),

               "test_mean_iou_epoch" : np.mean(iou_score),
               "test_mean_recall_epoch" : np.mean(recall_list),
               "test_mean_precision_epoch":np.mean(precision_list),
               "test_mean_sensitivity_epoch":np.mean(sensitivity_list),
               "test_mean_specificity_epoch":np.mean(specificity_list),
               "test_mean_f1_epoch":np.mean(f1_list),
               "test_mean_acc_epoch":np.mean(acc_list),

               "test_mean_iou_1" : np.mean(iou_score1),
               "test_mean_recall_1" : np.mean(recall_list1),
               "test_mean_precision_1":np.mean(precision_list1),
               "test_mean_sensitivity_1":np.mean(sensitivity_list1),
               "test_mean_specificity_1":np.mean(specificity_list1),
               "test_mean_f1_1":np.mean(f1_list1),
               "test_mean_acc_1":np.mean(acc_list1),

               "test_mean_iou_2" : np.mean(iou_score2),
               "test_mean_recall_2" : np.mean(recall_list2),
               "test_mean_precision_2":np.mean(precision_list2),
               "test_mean_sensitivity_2":np.mean(sensitivity_list2),
               "test_mean_specificity_2":np.mean(specificity_list2),
               "test_mean_f1_2":np.mean(f1_list2),
               "test_mean_acc_2":np.mean(acc_list2),

               "test_mean_iou_3" : np.mean(iou_score3),
               "test_mean_recall_3" : np.mean(recall_list3),
               "test_mean_precision_3":np.mean(precision_list3),
               "test_mean_sensitivity_3":np.mean(sensitivity_list3),
               "test_mean_specificity_3":np.mean(specificity_list3),
               "test_mean_f1_3":np.mean(f1_list3),
               "test_mean_acc_3":np.mean(acc_list3),

               "test_mean_iou_4" : np.mean(iou_score4),
               "test_mean_recall_4" : np.mean(recall_list4),
               "test_mean_precision_4":np.mean(precision_list4),
               "test_mean_sensitivity_4":np.mean(sensitivity_list4),
               "test_mean_specificity_4":np.mean(specificity_list4),
               "test_mean_f1_4":np.mean(f1_list4),
               "test_mean_acc_4":np.mean(acc_list4),

               "test_mean_iou_5" : np.mean(iou_score5),
               "test_mean_recall_5" : np.mean(recall_list5),
               "test_mean_precision_5":np.mean(precision_list5),
               "test_mean_sensitivity_5":np.mean(sensitivity_list5),
               "test_mean_specificity_5":np.mean(specificity_list5),
               "test_mean_f1_5":np.mean(f1_list5),
               "test_mean_acc_5":np.mean(acc_list5),

                })
    print("end")
    wandb.finish()

