
import pickle
import gzip
from PIL import Image
import scipy
import numpy as np
import keras
import os
from keras.layers import Dense
from keras.models import Sequential
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics
import operator



def preprocess():
    # ## Load MNIST on Python 3.x
    filename = 'mnist.pkl.gz'
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    
    # ## Load USPS on Python 3.x
    USPSMat  = []
    USPSTar  = []
    curPath  = 'USPSdata/Numerals'

    for j in range(0,10):
        curFolderPath = curPath + '/' + str(j)
        imgs =  os.listdir(curFolderPath)
        for img in imgs:
            curImg = curFolderPath + '/' + img
            if curImg[-3:] == 'png':
                img = Image.open(curImg,'r')
                img = img.resize((28, 28))
                imgdata = (255-np.array(img.getdata()))/255
                USPSMat.append(imgdata)
                USPSTar.append(j)
            
    return training_data,validation_data,test_data,USPSMat,USPSTar



def costFunction(design_matrix,tcap,weights,nrows):
    x=tcap.shape[0]
    len=design_matrix.shape[0]
    class1cost=-len*(np.log(tcap))
    class2cost=(1-len)*(np.log(1-tcap))
    cost=class1cost-class2cost
    cost=cost.sum()/x
    return cost
    


def oneHotIt(Y):
    m = Y.shape[0]
    OHX = scipy.sparse.csr_matrix((np.ones(m), (Y, np.array(range(m)))))
    OHX = np.array(OHX.todense()).T
    #print('one hot done')
    return OHX



def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    #print('softmax done')
    return sm



def Loss(w,x,y,l):
    m = x.shape[0] 
    y_mat = oneHotIt(y) 
    scores = np.dot(x,w) 
    prob = softmax(scores) 
    loss = (-1 / m) * np.sum(y_mat * np.log(prob)) + (l/2)*np.sum(w*w) 
    grad = (-1 / m) * np.dot(x.T,(y_mat - prob)) + l*w 
    return loss,grad



def Logistic(x,y):
    w = np.zeros([x.shape[1],len(np.unique(y))])
    l = 1
    nepochs= 300
    learning_Rate = 0.05
    cost = []
    for i in range(0,nepochs):
        loss,grad = Loss(w,x,y,l)
        print('For iteration {} Loss is {}'.format(i,loss))
        cost.append(loss)
        w = w - (learning_Rate * grad)
    plt.plot(cost)
    return loss,w



def Accuracy(someX,someY,w):
    prob,p = probabilityAndpredictions(someX,w)
    accuracy = sum(p == someY)/(float(len(someY)))
    return accuracy,p


def probabilityAndpredictions(someX,w):
    probs = softmax(np.dot(someX,w))
    preds = np.argmax(probs,axis=1)
    return probs,preds



def callLogistic(dtr,ttr,dtest,ttest,dval,tval,USPSMat,USPSTar):
    
    
    print('Logistic Regression ------------ Please wait')

    loss,w1=Logistic(dtr,ttr)
    accuracy,tcap=Accuracy(dtr,ttr,w1)
    print('MNIST Training accuracy is : {}'.format(accuracy))
    
    
    accuracy,tcap2=Accuracy(dtest,ttest,w1)
    print('MNIST Testing accuracy is : {}'.format(accuracy))
    print(confusion_matrix(ttest, tcap2))
    
    accuracy,tcap3=Accuracy(dval,tval,w1)
    print('\nMNIST Validation accuracy is : {}'.format(accuracy))
    
    
    accuracy,tcap4=Accuracy(USPSMat,USPSTar,w1)
    print('USPS accuracy is : {}'.format(accuracy))
    print(confusion_matrix(USPSTar, tcap4))
    
    
    return tcap2,tcap4




def NeuralNetwork(X_train,y_train,X_test,y_test,USPSMat,USPSTar):
    
    num_classes=10
    image_vector_size=28*28
    X_train = X_train.reshape(X_train.shape[0], image_vector_size)
    X_test = X_test.reshape(X_test.shape[0], image_vector_size)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    USPSMat = USPSMat.reshape(USPSMat.shape[0], image_vector_size)
    USPSTar = keras.utils.to_categorical(USPSTar, num_classes)
    print('Neural Network -------- Please wait 10 minutes')
    
    
    image_size =784
    model=Sequential()
    model.add(Dense(units=32, activation='sigmoid', input_shape=(image_size,)))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=128, epochs=100,verbose=True)
    
    loss,accuracy = model.evaluate(X_test, y_test)
    new=model.predict_classes(X_test)

    print("For MNIST \nAccuracy: {} and Loss: {} ".format(accuracy,loss))
    
    
    loss2,accuracy2=model.evaluate(USPSMat, USPSTar)
    new2=model.predict_classes(USPSMat)
    print("For USPS \nAccuracy: {} and Loss: {} ".format(accuracy2,loss2))
    
    
    return loss,accuracy,loss2,accuracy2,new,new2



def SVMandRandomForest(X_train,y_train,X_test,y_test,USPSMat,USPSTar):
    
    
    # SVM
    print('SVM ------- Please wait')
    #change parameters as per project pdf 
    classifier1 = SVC(kernel='linear', C=2, gamma =0.05);
    classifier1.fit(X_train, y_train)
    print('Training done')
    y_prediction = classifier1.predict(X_test)
    newtest=classifier1.predict(USPSMat)
    
    print(confusion_matrix(y_test, y_prediction))
    print("For MNIST Accuracy: ",metrics.accuracy_score(y_test, y_prediction))
    print(confusion_matrix(USPSTar, newtest))
    print("\nFor USPS Accuracy: ",metrics.accuracy_score(USPSTar, newtest))
    
    
    
    #RandomForestClassifier
    print('\nRandom Forest ------- Please wait')
    classifier2 = RandomForestClassifier(n_estimators=100);
    classifier2.fit(X_train, y_train)
    y_prediction2 = classifier2.predict(X_test)
    newtest2=classifier2.predict(USPSMat)
    
    print(confusion_matrix(y_test, y_prediction))
    print("For MNIST Accuracy Accuracy: \n",metrics.accuracy_score(y_test, y_prediction))
    print(confusion_matrix(USPSTar, newtest))
    print("For USPS Accuracy: ",metrics.accuracy_score(USPSTar, newtest))
    
    return y_prediction,newtest,y_prediction2,newtest2
    


def voting(log,log2,new,new2,svm,svm2,rf,rf2):
    n1=[]
    n2=[]
    
    for i in range(0,log.shape[0]):
        d={}
        d[log[i]]=1
        if(new[i] not in d):
            d[new[i]]=1
        else:
            y=d.get(new[i])
            y+=1
            d[new[i]]=y
            
        if(svm[i] not in d):
            d[svm[i]]=1
        else:
            y=d.get(svm[i])
            y+=1
            d[svm[i]]=y
            
        if(rf[i] not in d):
            d[rf[i]]=1
        else:
            y=d.get(rf[i])
            y+=1
            d[rf[i]]=y
        w=max(d.items(), key=operator.itemgetter(1))[0]
        n1.append(w)  
    
    
    for i in range(0,log2.shape[0]):
        
        d={}
        d[log2[i]]=1
        if(new2[i] not in d):
            d[new2[i]]=1
        else:
            y=d.get(new2[i])
            y+=1
            d[new2[i]]=y
            
        if(svm2[i] not in d):
            d[svm2[i]]=1
        else:
            y=d.get(svm2[i])
            y+=1
            d[svm2[i]]=y
            
        if(rf2[i] not in d):
            d[rf2[i]]=1
        else:
            y=d.get(rf2[i])
            y+=1
            d[rf2[i]]=y
        w=max(d.items(), key=operator.itemgetter(1))[0]
        n2.append(w)

    n1=np.array(n1)
    n2=np.array(n2)
    return n1,n2



def calcAccuracy(target,tcap):
    n=len(tcap)
    count=0
    for i in range(0,n):
        if(target[i]==tcap[i]):
            count+=1
    accuracy=float((count/n)*100)
    return accuracy




training_data,validation_data,test_data,USPSMat,USPSTar=preprocess()
X_train,y_train=training_data[0],training_data[1]
X_test,y_test=test_data[0],test_data[1]
X_val,y_val=validation_data[0],validation_data[1]
USPSMat=np.array(USPSMat)
USPSTar=np.array(USPSTar,dtype='int64')
print('Preprocessing Done \n')


log,log2=callLogistic(X_train,y_train,X_test,y_test,X_val,y_val,USPSMat,USPSTar)
loss,accuracy,loss2,accuracy2,new,new2=NeuralNetwork(X_train,y_train,X_test,y_test,USPSMat,USPSTar)
print('Confusion matrix for MNIST')
print(confusion_matrix(y_test, new))
print('\nConfusion matrix for USPS')
print(confusion_matrix(USPSTar, new2))
svm,svm2,rf,rf2=SVMandRandomForest(X_train,y_train,X_test,y_test,USPSMat,USPSTar)
a,b=voting(log,log2,new,new2,svm,svm2,rf,rf2)
acc1=calcAccuracy(y_test,a)
acc2=calcAccuracy(USPSTar,b) 
print('After voting {} {}' .format(acc1,acc2))