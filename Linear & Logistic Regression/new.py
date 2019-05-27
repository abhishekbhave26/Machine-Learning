# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 20:35:29 2018

@author: abhis
"""

import numpy as np
import pandas as pd
import csv
import math
import time
from pathlib import Path
import tensorflow as tf

start=time.time()

def createandappendSameandDiffPairs(samepair,diffpair):
    
    df = pd.DataFrame( columns = ['img_id_A', 'img_id_B', 'target']) 
    
    x=samepair.shape[0]
    if(x==71531):
        samepair=samepair.sample(10000)
        
    df = df.append(samepair, ignore_index = True)
    
    if(x==71531): 
        a=diffpair.sample(10000)
    else:
        a=diffpair.sample(791)
    df=df.append(a) 
    return df


def createHodConcatandSub():  
    
    print('Creating HOD Files Wait for 15 seconds')
    
    samepair=pd.read_csv('hodsame_pairs.csv')
    diffpair=pd.read_csv('hoddiffn_pairs.csv')
    new_df=createandappendSameandDiffPairs(samepair,diffpair)
    features=pd.read_csv('HumanObserved-Features-Data.csv')
    
    new_dfforsub=new_df
    x,y=new_df.shape
    p,q=features.shape
    
    dictfeatures=features.set_index('img_id').transpose().to_dict(orient='list')
    
    newList=[]
    newList2=[]
    for i in range(0,x):
        list=[]
        
        for j in range(0,y-1):
            dfvalue=new_df.iloc[i][j]
            a=dictfeatures[dfvalue]
            # for gsc comment below line
            a=a[1:]
            list.extend(a)
        
        list2=list
        pr=np.array(list2)
        p, r = pr[0: 9], pr[9: ]
        result=np.abs(p-r)
        
        q=pd.DataFrame(np.array(list).reshape(1,18), columns =list)
        d=np.array(q)
        newList.append(d)
        
        e=pd.DataFrame(np.array(result).reshape(1,9), columns =result)
        f=np.array(e)
        newList2.append(f)
        
        
    #for setting 1 and 2
    new_df=new_df.reset_index()
    new_dfforsub=new_dfforsub.reset_index()
    
    grt=pd.DataFrame(np.array(newList).reshape(1582,18), columns =list)
    grt2=pd.DataFrame(np.array(newList2).reshape(1582,9), columns =result)      
          
    result = new_df.join(grt)
    resultsub=new_dfforsub.join(grt2)
    
    
    result.to_csv('matrixhod1.csv')
    resultsub.to_csv('matrixhod2.csv')
    
    return result,resultsub



#for gsc
def createGscConcatandSub():
   
    print('Creating GSC Files Wait for 7 minutes')
    
    
    samepair=pd.read_csv('gscsame_pairs.csv')
    diffpair=pd.read_csv('gscdiffn_pairs.csv')
    new_df=createandappendSameandDiffPairs(samepair,diffpair)
    features=pd.read_csv('GSC-Features.csv')   
    
    new_dfforsub=new_df
    x,y=new_df.shape
    p,q=features.shape
    
    dictfeatures=features.set_index('img_id').transpose().to_dict(orient='list')
    
    newList=[]
    newList2=[]
    for i in range(0,x):
        list=[]
        
        for j in range(0,y-1):
            dfvalue=new_df.iloc[i][j]
            a=dictfeatures[dfvalue]
            list.extend(a)
        
        list2=list
        pr=np.array(list2)
        p, r = pr[0: 512], pr[512: ]
        result=np.abs(p-r)
        
        q=pd.DataFrame(np.array(list).reshape(1,1024), columns =list)
        d=np.array(q)
        newList.append(d)
        
        
        e=pd.DataFrame(np.array(result).reshape(1,512), columns =result)
        f=np.array(e)
        newList2.append(f)
        
        
    #for setting 1 and 2
    new_df=new_df.reset_index()
    new_dfforsub=new_dfforsub.reset_index()
    
    grt=pd.DataFrame(np.array(newList).reshape(20000,1024), columns =list)
    grt2=pd.DataFrame(np.array(newList2).reshape(20000,512), columns =result)      
          
    result = new_df.join(grt)
    resultsub=new_dfforsub.join(grt2)
    
    result.to_csv('matrixgsc1.csv')
    resultsub.to_csv('matrixgsc2.csv')

    return result,resultsub
    

def createtcap(W_Now,design_matrix):
    return np.dot(design_matrix,W_Now)


def createtcapforLogistic(W_Now,design_matrix):
    inside_part=np.dot(design_matrix,W_Now)
    #print(inside_part.shape)
    list=[]
    len=design_matrix.shape[0]
    for i in range(0,inside_part.shape[0]):
        sigmoid=1/(1+(math.exp(-inside_part[i])))
        list.append(sigmoid)
    sigmoid=np.array((list))
    sigmoid=np.array(sigmoid.reshape(len,1))
    #print(sigmoid)
    #print(sigmoid.shape)
    return sigmoid

    
def calcAccuracy(target,tcap):
    n=len(tcap)
    count=0
    for i in range(0,n):
        if(target[i]==tcap[i]):
            count+=1
    accuracy=float((count/n)*100)
    return accuracy
    

def gradientDescent(design_matrix,target,learning_rate,nepochs,nrows):

    W_After=np.array([])
    dms=design_matrix.shape[1]
    len=design_matrix.shape[0]
    #print(len,dms)
    W_Now=np.zeros(shape=[dms,1])
    tcap=np.empty([nrows,1])
    for i in range(0,nepochs): 
        tcap=np.array(createtcap(W_Now,design_matrix).reshape(nrows,1))
        error=target-tcap
        #print(delta_inside_part.shape)
        
        x=np.transpose(design_matrix)
        #print(x.shape)
        delta=np.dot(-x,error)
        #print(delta)
        #delta=np.transpose(delta)
        delta/=len
        W_After=W_Now-np.dot(delta,learning_rate)
        #print(W_After.shape)
        #print(W_After)
        
        W_Now=W_After
        
        erms=GetErms(tcap,target)
        #print(erms)
        
    return erms,tcap,W_Now

 
     
def classify(tcap):
    result=tcap
    len=tcap.shape[0]
    for i in range(0,len):
        if(tcap[i]>=0.5):
            result[i]=1
        else:
            result[i]=0
    return result
    

def logisticRegression(design_matrix,target,learning_rate,nepochs,nrows):

    W_After=np.array([])
    dms=design_matrix.shape[1]
    len=design_matrix.shape[0]
    W_Now=np.zeros(shape=[dms,1])
    tcap=np.empty([nrows,1])
    cost_history=[]
    for i in range(0,nepochs): 
        tcap=np.array(createtcapforLogistic(W_Now,design_matrix).reshape(nrows,1))
        error=target-tcap
        #print(delta_inside_part.shape)
        x=np.transpose(design_matrix)
        #print(x.shape)
        delta=np.dot(-x,error)
        #print(delta)
        #delta=np.transpose(delta)
        delta/=len
        W_After=W_Now-np.dot(delta,learning_rate)
        #print(W_After.shape)
        #print(W_After)
        
        cost=costFunction(design_matrix,tcap,W_Now,len)
        cost_history.append(cost)
        #print('Iteration : {} and cost is :{}'.format(i,cost))
        
        W_Now=W_After
        
    return W_Now,cost_history,tcap
   


def takeinput(hodconcat,hodsub,gscconcat,gscsub):
    print('Enter 1 for HOD Concatenation')
    print('Enter 2 for HOD Subtraction')
    print('Enter 3 for GSC Concatenation')
    print('Enter 4 for GSC Subtraction')
    
    switchNumber=int(input('Enter as above: '))
    if(switchNumber>4 or switchNumber<1):
        print('Wrong Input,Enter Again :')
        takeinput(hodconcat,hodsub,gscconcat,gscsub)
        
    elif(switchNumber==1):
        #hod concat
        design_matrix=np.array(hodconcat.iloc[:,5:])
        target=np.array(hodconcat.iloc[:,4:5], dtype='float64')
        
        dtr,dtest,dval=partition(design_matrix)
        ttr,ttest,tval=partition(target)
        
       
        print('Enter 1 for Linear Regression')
        print('Enter 2 for Logistic Regrssion')
        choice=int(input('Enter as above : '))
        if(choice==1):
            
            learning_rate=0.005
            nepochs=3000
            
            erms,tcap,W=gradientDescent(dtr,ttr,learning_rate,nepochs,dtr.shape[0])
            print('Training Erms is : {} '.format(erms))
            #print ("Learning Rate eta ={} and epochs is {}".format(learning_rate,nepochs))
            
            erms,tcap,W=gradientDescent(dtest,ttest,learning_rate,nepochs,dtest.shape[0])
            print('Testing Erms is : {} '.format(erms))
            #print ("Learning Rate eta ={} and epochs is {}".format(learning_rate,nepochs))
            
            erms,tcap,W=gradientDescent(dval,tval,learning_rate,nepochs,dval.shape[0])
            print('Validation Erms is : {} '.format(erms))
            print ("Learning Rate eta ={} and epochs is {}".format(learning_rate,nepochs))
        
        elif(choice==2):
            
            learning_rate=0.03
            nepochs=2000
            
            W_Now,cost_history,tcap=logisticRegression(dtr,ttr,learning_rate,nepochs,dtr.shape[0])
            tcap=classify(tcap)
            print('Training Accuracy is {}'.format(calcAccuracy(target,tcap)))
            
            W_Now,cost_history,tcap=logisticRegression(dtest,ttest,learning_rate,nepochs,dtest.shape[0])
            tcap=classify(tcap)
            print('Testing Accuracy is {}'.format(calcAccuracy(target,tcap)))
            
            W_Now,cost_history,tcap=logisticRegression(dval,tval,learning_rate,nepochs,dval.shape[0])
            tcap=classify(tcap)
            print('Validation Accuracy is {}'.format(calcAccuracy(target,tcap)))
            print ("Learning Rate eta ={} and epochs is {}".format(learning_rate,nepochs))
        else:
            pass
            #call neural network here
        return tcap,target
        
    
    elif(switchNumber==2):
        design_matrix=np.array(hodsub.iloc[:,5:])
        target=np.array(hodsub.iloc[:,4:5], dtype='float64')
        
        dtr,dtest,dval=partition(design_matrix)
        ttr,ttest,tval=partition(target)
        
        #call partition here
        print('Enter 1 for Linear Regression')
        print('Enter 2 for Logistic Regrssion')
        choice=int(input('Enter as above : '))
        if(choice==1):
            learning_rate=0.005
            nepochs=3000
            
            erms,tcap,W=gradientDescent(dtr,ttr,learning_rate,nepochs,dtr.shape[0])
            print('Training Erms is : {} '.format(erms))
            #print ("Learning Rate eta ={} and epochs is {}".format(learning_rate,nepochs))
            
            erms,tcap,W=gradientDescent(dtest,ttest,learning_rate,nepochs,dtest.shape[0])
            print('Testing Erms is : {} '.format(erms))
            #print ("Learning Rate eta ={} and epochs is {}".format(learning_rate,nepochs))
            
            erms,tcap,W=gradientDescent(dval,tval,learning_rate,nepochs,dval.shape[0])
            print('Validation Erms is : {} '.format(erms))
            print ("Learning Rate eta ={} and epochs is {}".format(learning_rate,nepochs))
            
            
            
        elif(choice==2):
            learning_rate=0.03
            nepochs=1000
            
            W_Now,cost_history,tcap=logisticRegression(dtr,ttr,learning_rate,nepochs,dtr.shape[0])
            tcap=classify(tcap)
            print('Training Accuracy is {}'.format(calcAccuracy(target,tcap)))
            
            W_Now,cost_history,tcap=logisticRegression(dtest,ttest,learning_rate,nepochs,dtest.shape[0])
            tcap=classify(tcap)
            print('Testing Accuracy is {}'.format(calcAccuracy(target,tcap)))
            
            W_Now,cost_history,tcap=logisticRegression(dval,tval,learning_rate,nepochs,dval.shape[0])
            tcap=classify(tcap)
            print('Validation Accuracy is {}'.format(calcAccuracy(target,tcap)))
            print ("Learning Rate eta ={} and epochs is {}".format(learning_rate,nepochs))
            
        
        else:
            pass
            #call neural network here
        
        
        return tcap,target
        
    
    elif(switchNumber==3):
        design_matrix=np.array(gscconcat.iloc[:,5:])
        target=np.array(gscconcat.iloc[:,4:5], dtype='float64')
        
        dtr,dtest,dval=partition(design_matrix)
        ttr,ttest,tval=partition(target)
        
        #call partition here
        print('Enter 1 for Linear Regression')
        print('Enter 2 for Logistic Regrssion')
        choice=int(input('Enter as above : '))
        if(choice==1):
            learning_rate=0.005
            nepochs=300
            
            print('Please wait for 15 minutes')
            erms,tcap,W=gradientDescent(dtr,ttr,learning_rate,nepochs,dtr.shape[0])
            print('Training Erms is : {} '.format(erms))
            #print ("Learning Rate eta ={} and epochs is {}".format(learning_rate,nepochs))
            
            erms,tcap,W=gradientDescent(dtest,ttest,learning_rate,nepochs,dtest.shape[0])
            print('Testing Erms is : {} '.format(erms))
            #print ("Learning Rate eta ={} and epochs is {}".format(learning_rate,nepochs))
            
            erms,tcap,W=gradientDescent(dval,tval,learning_rate,nepochs,dval.shape[0])
            print('Validation Erms is : {} '.format(erms))
            print ("Learning Rate eta ={} and epochs is {}".format(learning_rate,nepochs))
            
        elif(choice==2):
            
            learning_rate=0.005
            nepochs=300
            
            print('Please wait for 10 minutes')
            W_Now,cost_history,tcap=logisticRegression(dtr,ttr,learning_rate,nepochs,dtr.shape[0])
            tcap=classify(tcap)
            print('Training Accuracy is {}'.format(calcAccuracy(target,tcap)))
            
            W_Now,cost_history,tcap=logisticRegression(dtest,ttest,learning_rate,nepochs,dtest.shape[0])
            tcap=classify(tcap)
            print('Testing Accuracy is {}'.format(calcAccuracy(target,tcap)))
            
            W_Now,cost_history,tcap=logisticRegression(dval,tval,learning_rate,nepochs,dval.shape[0])
            tcap=classify(tcap)
            print('Validation Accuracy is {}'.format(calcAccuracy(target,tcap)))
            print ("Learning Rate eta ={} and epochs is {}".format(learning_rate,nepochs))
            
        else:
            pass
            #call neural network here
        return tcap,target
      
        
    elif(switchNumber==4):
        design_matrix=np.array(gscsub.iloc[:,5:])
        target=np.array(gscsub.iloc[:,4:5], dtype='float64')
        
        dtr,dtest,dval=partition(design_matrix)
        ttr,ttest,tval=partition(target)
        #call partition here
        print('Enter 1 for Linear Regression')
        print('Enter 2 for Logistic Regrssion')
    
        choice=int(input('Enter as above : '))
        if(choice==1):
            learning_rate=0.005
            nepochs=300
            
            print('Please wait for 15 minutes')
            erms,tcap,W=gradientDescent(dtr,ttr,learning_rate,nepochs,dtr.shape[0])
            print('Training Erms is : {} '.format(erms))
            #print ("Learning Rate eta ={} and epochs is {}".format(learning_rate,nepochs))
            
            erms,tcap,W=gradientDescent(dtest,ttest,learning_rate,nepochs,dtest.shape[0])
            print('Testing Erms is : {} '.format(erms))
            #print ("Learning Rate eta ={} and epochs is {}".format(learning_rate,nepochs))
            
            erms,tcap,W=gradientDescent(dval,tval,learning_rate,nepochs,dval.shape[0])
            print('Validation Erms is : {} '.format(erms))
            print ("Learning Rate eta ={} and epochs is {}".format(learning_rate,nepochs))
            
        elif(choice==2):
            learning_rate=0.05
            nepochs=300
            
            print('Please wait for 10 minutes')
            W_Now,cost_history,tcap=logisticRegression(dtr,ttr,learning_rate,nepochs,dtr.shape[0])
            tcap=classify(tcap)
            print('Training Accuracy is {}'.format(calcAccuracy(target,tcap)))
            
            W_Now,cost_history,tcap=logisticRegression(dtest,ttest,learning_rate,nepochs,dtest.shape[0])
            tcap=classify(tcap)
            print('Testing Accuracy is {}'.format(calcAccuracy(target,tcap)))
            
            W_Now,cost_history,tcap=logisticRegression(dval,tval,learning_rate,nepochs,dval.shape[0])
            tcap=classify(tcap)
            print('Validation Accuracy is {}'.format(calcAccuracy(target,tcap)))
            print ("Learning Rate eta ={} and epochs is {}".format(learning_rate,nepochs))
        else:
            pass
        #call neural network here
        
        return tcap,target
        
    else:
        return



def readfile():
#checking if hod concat and hod sub already present
    my_file = Path("matrixhod1.csv")
    my_file2= Path("matrixhod2.csv")
        
    if(my_file.is_file() and my_file2.is_file()) :
        hodconcat=pd.read_csv('matrixhod1.csv')
        hodsub=pd.read_csv('matrixhod2.csv')
        print('File Found')
    else:
        hodconcat,hodsub=createHodConcatandSub()
        hodconcat=pd.read_csv('matrixhod1.csv')
        hodsub=pd.read_csv('matrixhod2.csv')
        
        
    #checking if gsc concat and gsc sub already present
    my_file = Path("matrixgsc1.csv")
    my_file2= Path("matrixgsc2.csv")
        
    if(my_file.is_file() and my_file2.is_file()) :
        gscconcat=pd.read_csv('matrixgsc1.csv')
        gscsub=pd.read_csv('matrixgsc2.csv')
        print('File Found')
    else:
        gscconcat,gscsub=createGscConcatandSub()
        gscconcat=pd.read_csv('matrixgsc1.csv')
        gscsub=pd.read_csv('matrixgsc2.csv')
    
    return hodconcat,hodsub,gscconcat,gscsub



def GetErms(tcap,target):
    sum = 0.0
    accuracy = 0.0
    counter = 0

    for i in range (0,len(tcap)):
        sum = sum + math.pow((target[i] - tcap[i]),2)
        if(int(np.around(tcap[i], 0)) == target[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(tcap)))

    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(tcap))))


def costFunction(design_matrix,tcap,weights,nrows):
    x=tcap.shape[0]
    #tcap=createtcap(weights,design_matrix)
    #tcap=np.array(createtcapforLogistic(weights,design_matrix).reshape(nrows,1))
    # when label=1
    len=design_matrix.shape[0]
    class1cost=-len*(np.log(tcap))
    #when label=1
    class2cost=(1-len)*(np.log(1-tcap))
    cost=class1cost-class2cost
    cost=cost.sum()/x
    return cost



def partition(x):
    #print(x.shape)
    a=x.shape[0]
    np.random.shuffle(x)
    p=int((80*a/100))
    q=int(a*1/10)
    training, test,val = x[:p], x[p:p+q],x[p+q:]
    return training,test,val




hodconcat,hodsub,gscconcat,gscsub=readfile()
tcap,target=takeinput(hodconcat,hodsub,gscconcat,gscsub)





end=time.time()
#print('Time taken: {} minutes'.format((end-start)/60)) 