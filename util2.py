from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2 as cv
import os


# 讀出mnist內建的檔案 並做一些調整
def mnist_data():
    (train_images,train_labels),(test_images,test_labels)=mnist.load_data()
    x_train=train_images.reshape((60000,28*28)) # 將(60000, 28, 28) 轉換成(60000,784)
    x_train=x_train.astype('float32')/255 # 再將 0~255的像素轉換成 0~1的浮點數

    x_test=test_images.reshape((10000,28*28)) # 對測試資料做同樣的轉換
    x_test=x_test.astype('float32')/255

    y_train=to_categorical(train_labels) # 把標籤從數字變成0 0 0 0 1 0 0 0 0 0
    y_test=to_categorical(test_labels)
    return (x_train, x_test), (y_train, y_test)


# 建一個兩層的model
def mnist_model():
    model=Sequential()
    model.add(Dense(512, activation='relu', input_dim= 784)) # 加入第一層
    model.add(Dense(10, activation='softmax')) # 加入第二層
    model.summary()
    model.compile(optimizer='rmsprop', # 指定優化器
                    loss='categorical_crossentropy', # 指定損失函數
                    metrics=['acc']) # 指定評量準則
    return model


# 印出線條
def plot(history_dict, keys, title=None, xyLabel=[], ylim=(), size=()):
    lineType=('-', '--', '.') # 線條的樣式 畫多條線時會依序採用
    if len(ylim)==2: plt.ylim(*ylim) # 設定y軸最小值及最大值
    if len(size)==2: plt.gcf().set_size_inches(*size) 
    epochs=range(1,len(history_dict[keys[0]])+1) # 計算有幾周期的資料
    for i in range(len(keys)): # 走訪每一個key 例如 loss acc等
        plt.plot(epochs, history_dict[keys[i]], lineType) # 畫出線條
    if title: # 是否顯示標題
        plt.title(title)
    if len(xyLabel)==2: # 是否顯示x,y軸的說明文字
        plt.xlabel(xyLabel[0])
        plt.ylabel(xyLabel[1])
    plt.legend(keys, loc='best') # 顯示圖例
    plt.show()


# 讀取原始圖片及標籤 並轉為灰階
# 標籤取檔名的_前的字串
def load_origin(imgs_dir,pre_adjust):
    filenames=os.listdir(imgs_dir)
    labels=[]
    imgs=[]
    for filename in filenames:
        full_path=os.path.join(imgs_dir,filename)
        img = cv.imread(full_path)
        img  = cv.cvtColor(img , cv.COLOR_BGR2GRAY)
        if pre_adjust:
            img=adjust_img(img)
        imgs.append(img)

        label=filename.split('_')
        labels.append(label[0])
    return imgs,labels


# 用load_origin讀取圖片和標籤後 若adjust為True則用adjust_img對影像進行調整
def load_dataset(imgs_dir,pre_adjust):
    filenames=os.listdir(imgs_dir)
    imgs,labels=load_origin(imgs_dir,pre_adjust)

    imgs=np.array(imgs)
    imgs=imgs.reshape((len(filenames),28*28)) # 對測試資料做同樣的轉換
    imgs=imgs.astype('float32')/255    
    labels=to_categorical(labels) # 把標籤從數字變成0 0 0 0 1 0 0 0 0 0
    return imgs,labels


# 調整圖片 使其容易辨識
def adjust_img(img):
    img=move_to_center(img)
    return img


# 將圖片中的內容移至中心
def move_to_center(img):
    centerY=round(cv.moments(img)['m01']/np.sum(img))
    centerX=round(cv.moments(img)['m10']/np.sum(img))
    centerY=round(centerY)
    centerX=round(centerX)

    # 平移矩陣
    m=np.float32([[1,0,img.shape[1]/2-centerX],
                [0,1,img.shape[1]/2-centerY]])
    img=cv.warpAffine(img,m,(img.shape[1],img.shape[0]))
    return img