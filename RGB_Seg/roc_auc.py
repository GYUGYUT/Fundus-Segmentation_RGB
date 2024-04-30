import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
# 예측값 (predicted values)과 실제값 (true values)을 준비합니다.
def roc_auc_save(y_true, y_pred,args):
    # ROC 곡선을 계산합니다.
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    filepath = os.path.join(args.save_path_conpusion, '{}_plot_roc_auc.png'.format(args.arch))
    # AUC를 계산합니다.
    roc_auc = auc(fpr, tpr)

    # ROC 곡선을 그립니다.
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(filepath)  