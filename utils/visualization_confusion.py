import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.confusion import confusion

def visualization_confusion(loader_test,prediction):

    #the label of testing dataset
    label = np.empty(0,)
    for i in range(len(loader_test.dataset)):
        label = np.append(label,loader_test.dataset[i].y)

    #confusion matrix
    confusion_data = confusion_matrix(label,prediction)
    print("="*50)
    print("当前混淆矩阵分类：")
    print(confusion_data.shape[0])
    print("=" * 50)
    if confusion_data.shape[0]==12:
        confusion(confusion_matrix=confusion_data)  #混淆矩阵绘制
    else:
        raise ValueError('666')



