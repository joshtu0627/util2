from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import numpy as np
import tensorflow as tf
import util2 as u

# 讀取訓練好的模型
model=tf.keras.models.load_model('MnistModel.h5')

# 用util2讀取檔案
my_images,my_labels=u.load_dataset('data',True)
origin_images,origin_labels=u.load_origin('data',True)

# 用模型預測答案
predict=np.argmax(model.predict(my_images), axis=-1) # 回傳預測結果中機率最高的索引位置 也就是如果是 0 0 0 1 0 0 0 0 0 0則會回傳3
predict.round(1) # 四捨五入至整數

# 印出測試圖片並標示預測結果及答案
plt.gcf().set_size_inches(15,7) # 設定圖片的寬和高(英吋)
for i in range(len(origin_images)):
    ax=plt.subplot(int(len(origin_images)/5+1), 5,1+i) # 設定行數為圖片數量/5的子圖表 目前要畫第1+i個
    ax.imshow(origin_images[i],cmap='gray') # 顯示灰階圖片(黑底白字)
    ax.set_title('label='+str(origin_labels[i])+'\npredict='+str(predict[i]),fontsize=18)
    ax.set_xticks([]);ax.set_yticks([]) # X,Y軸不顯示刻度
plt.show()