# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 22:55:11 2018

@author: manis
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 22:40:49 2018

@author: manis
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 22:19:29 2018

@author: Nirjhar Roys
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:50:22 2018


"""
import csv
import numpy as np
from numpy import empty
from numpy import array
from scipy import sparse
from sklearn.cluster import KMeans
import matplotlib.pyplot as matplot
from sklearn.random_projection import SparseRandomProjection
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.metrics.pairwise import pairwise_distances
import random
import timeit




class Node:

    def __init__(self, k,row_sample_count):

        self._child = []
        self._k = k
        self.isLeaf=0
        self.classif=[]
        self.classif_original_dimension=[]
        self.y_mean=0
        self.train_data_sampled_total=[]
        self.row_sample_count=row_sample_count
    
    def testStopCondition(self,train_data_X, train_data_Y,limit ):
        
        #if(len(train_data_X)<self._k):#nof of samples less than no ofclusters , then we should stop
           # return 1;
        
        #if(limit<0):
           # return 1;
        #return 0
        #print(len(train_data_X))
        #print(len(train_data_Y))
        
        if(len(train_data_X) < 200):
            return 1
        if(np.isclose(train_data_X, train_data_X[0]).all()):
            return 1
        elif((train_data_Y == train_data_Y[0]).all()):
            return 1
       
        else:
            return 0
        
        
        
        
    
    
    
        
    def computeMeanLabelVector(self,train_data_Y):
        ymean=  np.array(train_data_Y).mean(0)
        return ymean
    
    
    
    
    def computeCentroid(self,train_data_X_sampled,labels_,i):
        a=[]        
        kk=0;        
        for l in labels_:
            if(l==i):                
                a.append(train_data_X_sampled[kk])
            kk=kk+1  
        
        a=np.array(a)        
        return a.mean(0)
            
        
        
    
    
    
    
    
    
    def trainNodeClassifier(self,train_data_X, train_data_Y,new_feature_count_list ):
        
        
        
        train_data_X=np.array(train_data_X)
        train_data_Y=np.array(train_data_Y)
        
        #--------------------------------sampling of X AND Y-------

        
        l=0
        label_count=train_data_Y.shape[1]
        feature_count=train_data_X.shape[1]
        #for l in range (0,3):
    
        new_feature_count  = (int)(new_feature_count_list * feature_count)
        transformer = SparseRandomProjection(n_components=new_feature_count, density='auto', eps=0.5, dense_output=False, random_state=None)
        
        #print('Size of total data at present node ',train_data_X.shape)
        train_data_X_sampled = transformer.fit_transform(train_data_X)
        
        
        #self.train_data_sampled_total=train_data_X_sampled
        
        #NOW SELECT A  SUBSET OF train_data_X_sampled (sampling of rows) 
        row_count_X = (int)(len(train_data_X))
        sample_row_count_X = (int)(row_count_X  * self.row_sample_count )
        random_index= random.sample(range(0, sample_row_count_X), sample_row_count_X )
        
        X_data_row_sample = empty((sample_row_count_X,new_feature_count))
        #print(row_count_X)
        #print(sample_row_count_X)
        #print(random_index)
        a=0
        c=0
        for a in random_index:
            X_data_row_sample[c,:] = train_data_X_sampled[a,:].copy()
            c=c+1
        self.train_data_sampled_total=train_data_X_sampled
        
                                      
        
        
        
        
        
    
        new_label_count  = (int)(new_feature_count_list * label_count)
        transformer = SparseRandomProjection(n_components=new_label_count, density='auto', eps=0.5, dense_output=False, random_state=None)
        train_data_Y_sampled = transformer.fit_transform(train_data_Y)    
        #NOW SELECT A  SUBSET OF train_data_y_sampled (sampling of rows) and feed it into KMEANs
    
        row_count_Y = (int)(len(train_data_Y))
        sample_row_count_Y = sample_row_count_X
        #Y_random_index= random.sample(range(0, row_count_X), sample_row_count_X )
        
        Y_data_row_sample = empty((sample_row_count_Y,new_label_count))
        c=0
        a=0
        for a in random_index:
            Y_data_row_sample[c,:] = train_data_Y_sampled[a,:].copy()
            c=c+1
        #self.train_data_sampled_total=X_data_row_sample
    
        

        #--------- clustering----------------------------------------------------------------------------------------------------
    
        k=self._k 
        cost = empty((3))   
        kmeans = KMeans(n_clusters=k, random_state=1)
        clusters = kmeans.fit_predict( Y_data_row_sample)
        centers = np.array(kmeans.cluster_centers_)
        cost[l] = kmeans.inertia_    
        #print("k: ",k,"feature count : ",new_feature_count,"cost: ",cost[l])
        
        
        classiff=[]
        for i in range(0,self._k):
            
            classiff_i=self.computeCentroid(X_data_row_sample,clusters,i)
            classiff.append(classiff_i)
        
        
        
        classiff=np.array(classiff)
        
        
        return classiff
            
        
        
        
    
    
    
    def split(self,classif, train_data_X, train_data_Y ,i):
        
        
        partition_x=[]
        partition_y=[]     
        
        kk=0
        
        for x in self.train_data_sampled_total :
        
            pairwise_distances_test_i=pairwise_distances(classif,[x])    
            predicted_class_array_index=np.argsort(  pairwise_distances_test_i.flatten()    )[0:1] #getting the predicted class          
                    
            if(predicted_class_array_index==i):
                partition_x.append(train_data_X[kk])
                partition_y.append(train_data_Y[kk])
            kk=kk+1     
        
        return partition_x,partition_y 
    
    
    
    
    
    def trainTree(self, train_data_X,train_data_Y,limit,new_feature_count_list):       
        
        self.isLeaf=self.testStopCondition(train_data_X, train_data_Y,limit )
        
        if(self.isLeaf==0):    
            
            self.classif=self.trainNodeClassifier(train_data_X, train_data_Y,new_feature_count_list )
            self.classif=np.array(self.classif)          
            
            X_child=[]
            Y_child=[]

            for i in range(0,self._k):
                X_ci , Y_ci=self.split(self.classif, train_data_X, train_data_Y ,i)
                #print('Child '+str(i)+' at level  '+  str(limit)+' of size '    +  str(len(X_ci)))
                X_child.append(X_ci)
                Y_child.append(Y_ci)
                
                self.classif_original_dimension.append(np.array(X_ci).mean(0))
                
            
            
            self.classif_original_dimension=np.array(self.classif_original_dimension)
            for i in range(0,self._k):
                self._child.append(Node(self._k,self.row_sample_count))
            
            for i in range(0,self._k):
                
                self._child[i].trainTree(X_child[i],Y_child[i],limit-1,new_feature_count_list)
        else:
            if(np.shape(train_data_Y)[0]>0):
                self.y_mean=self.computeMeanLabelVector(train_data_Y)
            else:
                self.y_mean=np.zeros(983)
            
            
            
            
        
        
        
    def testing(self,test_x):
            
        if(self.isLeaf==0):
                
            pairwise_distances_test_i=pairwise_distances(self.classif_original_dimension,[test_x])    
            predicted_class_array_index=np.argsort(  pairwise_distances_test_i.flatten()    )[0:1] #getting the predicted class 
            return self._child[predicted_class_array_index[0]].testing(test_x)
             
        else:
            return self.y_mean
             
                
            
            

        
        

def get_error(y_original, y_predicted,p):
    
    y_original=np.array(y_original)
    y_predicted=np.array(y_predicted)
    #pairwise_distances_test_i=pairwise_distances(self.classif_original_dimension,[test_x])    
    predicted_y_indices=np.argsort(  y_predicted.flatten()    )[-p:]
    
    original_y_indices=[]
    index_original=0
    for i in y_original:
        if(i==1):
            original_y_indices.append(index_original)  
        index_original=index_original+1
    
    
    original_y_indices=np.array(original_y_indices)
        
    
    
        
    
        
    #original_y_indices=np.argsort(  y_original.flatten()    )[-p:]
        
    #print('for original ',original_y_indices)
    #print('for predicted ',predicted_y_indices)
    
    no_of_common_indices=np.intersect1d(predicted_y_indices,original_y_indices).size
    #print(no_of_common_indices)
    
    error=no_of_common_indices/p
    
    #print(error)
    
    return error              
        
        


total_data_count=0
label_perfect_data_count=0
label_missing_data_count=0
feature_count=0
label_count=0
i=0
j=0
k=0
l=0
read_line = 1
label_length=0


#--------------read data file to get dimensions-------------------------------------------------------------------------
with open('Delicious_data.txt','r') as csv_file:
    file_reader = csv.reader(csv_file, delimiter=' ');
    
    for line in file_reader:
        if read_line == 1:
            total_data_count =(int) (line[0])
            feature_count = (int)(line[1])
            label_count  = (int)(line[2])
            read_line =2
        else :
            if len(line[0]) == 0 :
                label_missing_data_count =(int) (label_missing_data_count) + 1
            else :
                label_perfect_data_count =(int) (label_perfect_data_count ) + 1


total_X = empty((total_data_count,feature_count))
total_Y =empty((total_data_count,label_count))
total_Y_label =np.zeros((total_data_count,label_count))

read_line = 1
label_length=0
i=0
j=0
k=0
l=0
data_count =0

flag =1

with open('Delicious_data.txt','r') as csv_file:
    file_reader = csv.reader(csv_file, delimiter=' ');
    
    for line in file_reader:
        if read_line == 1 :
            read_line =2
            
        else :
            flag =1
            for j in range (0,feature_count) :
                if flag ==1 :
                    try :
                #print("line [0]",line[0])
                #print("line 1", line[1])
                        feature_index = (int)(line[j+1].partition(":")[0])
                        feature=(float)(line[j+1].partition(":")[2])
                #val=format('{0:.6f}',(line[j+1].partition(":")[2]))
                    except IndexError :
                        flag=0
                               
                    total_X[data_count][feature_index] = feature
                
                
               
            if len(line[0]) == 0 :
                for label_length in range (0,len(labels)) :
                    
                    total_Y[data_count][label_length +1] =0               
            else:
                labels=line[0].split(',')
               
                for label_length in range (0,len(labels)) :
                    
                    label = labels[label_length]
                    total_Y[data_count][label_length] =label
                   
                
           
            data_count =(int) (data_count) + 1

                        
a=0
b=0
for a in range (0,total_data_count):
    for b in range (0,label_count):
        if (total_Y[a][b] != 0):
            index= (int)(total_Y[a][b])
            total_Y_label[a][index]=1

#---------------TRAIN SPLIT data formation to craete TRAIN DATA SET & VALIDATION DATA SET-------------------------------

train_data_count = 0
validation_data_count=0
total_count=0
flag_train = 1
flag_validation=0

with open('delicious_trSplit.txt','r') as csv_file:
    file_reader = csv.reader(csv_file, delimiter=' ');
    
    for line in file_reader :
        #print(line[0])
        if len(line[0]) != 0 :
            total_count =total_count +1
        #if len(line[1]) != 0 : #AA
            #validation_data_count = validation_data_count +1    #AA
        
        

train_data_count = (int)(total_count * 0.6)
validation_data_count= (int)(total_count * 0.4)
train_data_indices = empty((train_data_count))
train_data_counter =0
validation_data_indices = empty((validation_data_count))
validation_data_counter =0

flag_train = 1
flag_validation=0


with open('delicious_trSplit.txt','r') as csv_file:
    file_reader = csv.reader(csv_file, delimiter=' ')
    
    for line in file_reader :
        #print(line[0])
        if len(line[0]) != 0 :
            if train_data_counter   < train_data_count  and flag_train ==1 :
                train_data_indices[train_data_counter] = (int)(line[0]) -1
                train_data_counter = train_data_counter + 1
                if train_data_counter == train_data_count :
                    flag_validation =1
                    flag_train =0
                    
            if validation_data_counter < (validation_data_count ) and flag_validation == 1 :
                validation_data_indices[validation_data_counter] = (int)(line[0]) -1
                validation_data_counter = validation_data_counter + 1  
                
        '''
        if len(line[1]) != 0 :
            validation_data_indices[validation_data_counter] = (int)(line[1]) -1
            validation_data_counter = validation_data_counter + 1  
        '''
            



train_data_X = empty((train_data_counter,feature_count))
train_data_Y = empty((train_data_counter,label_count))
validation_data_X = empty((validation_data_counter,feature_count))
validation_data_Y = empty((validation_data_counter,label_count))



train_counter = 0
validation_counter =0 
train_data_index = 0
validation_data_index = 0
for train_counter in range (train_data_counter) :
    train_counter = (int)(train_counter)
    train_data_index = (int)(train_data_indices[train_counter])
    train_data_X[train_counter, :] = total_X[train_data_index, :].copy()
    train_data_Y[train_counter, :] = total_Y_label[train_data_index, :].copy()
    #print(train_data_index)
    #print(total_X[train_data_index, :])
    
    
for validation_counter in range (validation_data_counter) :
    validation_counter = (int)(validation_counter)
    validation_data_index = (int)(validation_data_indices[validation_counter])
    validation_data_X[validation_counter, :] = total_X[validation_data_index, :].copy()
    validation_data_Y[validation_counter, :] = total_Y_label[validation_data_index, :].copy()





test_data_count = 0
validation_data_count=0



with open('delicious_tstSplit.txt','r') as csv_file:
    file_reader = csv.reader(csv_file, delimiter=' ');
    
    for line in file_reader :
        #print(line[0])
        if len(line[0]) != 0 :
            test_data_count = test_data_count +1
        
        
        



test_data_indices = empty((test_data_count))
test_data_counter =0
validation_data_indices = empty((test_data_count))
validation_data_counter =0

with open('delicious_tstSplit.txt','r') as csv_file:
    file_reader = csv.reader(csv_file, delimiter=' ');
    
    for line in file_reader :
        #print(line[0])
        if len(line[0]) != 0 :
            test_data_indices[test_data_counter] = (int)(line[0]) -1
            test_data_counter = test_data_counter + 1
                 
            



test_data_X = empty((test_data_counter,feature_count))
test_data_Y = empty((test_data_counter,label_count))
#validation_data_X = empty((validation_data_counter,feature_count))
#validation_data_Y = empty((validation_data_counter,label_count))



train_counter = 0
validation_counter =0 
train_data_index = 0
validation_data_index = 0
for test_counter in range (test_data_counter) :
    test_counter = (int)(test_counter)
    test_data_index = (int)(test_data_indices[test_counter])
    test_data_X[test_counter, :] = total_X[test_data_index, :].copy()
    test_data_Y[test_counter, :] = total_Y_label[test_data_index, :].copy()
    #print(train_data_index)
    #print(total_X[train_data_index, :])
    
    








#print(test_data_X,[2])











no_of_trees=1




forest=[]#WILL STORE COLLECTION OF FORESTS

#NOW WE WILL MANUALLY CREATE MANY TREEE  AND STORE THEM IN 'forest'


#forest.append(root)



'''
root = Node(5)
limit=3
new_feature_count_list=0.1
root.trainTree(train_data_X,train_data_Y,limit,new_feature_count_list)
forest.append(root)

'''

'''
root = Node(20)
limit=5
new_feature_count_list=0.25
root.trainTree(train_data_X,train_data_Y,limit,new_feature_count_list)
forest.append(root)
'''























    
print('\n\nTESTING AND TRAINING HAVE BEGUN\n')
    
limit=3


for k in [9]:
    
    start = timeit.default_timer()
    root = Node(k,0.6)
    root.trainTree(train_data_X,train_data_Y,limit,0.7)
    stop = timeit.default_timer()
    print('Time: to train tree 1 for k='+str(k), (stop - start))   
    
    
    start = timeit.default_timer()
    root2 = Node(k,0.6)
    root2.trainTree(train_data_X,train_data_Y,limit,0.6)
    stop = timeit.default_timer()
    print('Time: to train tree 2 for k='+str(k), (stop - start))      
    
    
    start = timeit.default_timer()
    root3 = Node(k,0.6)
    root3.trainTree(train_data_X,train_data_Y,limit,0.7)
    stop = timeit.default_timer()
    print('Time: to train tree 3 for k='+str(k), (stop - start))      
    
    start = timeit.default_timer()  
    root4 = Node(k,0.6)
    root4.trainTree(train_data_X,train_data_Y,limit,0.6)
    stop = timeit.default_timer()
    print('Time: to train tree 4 for k='+str(k), (stop - start))  
    
    
    
    predicted_list_y=[]
    for test_x in validation_data_X :
        test_y_predicted_tree1=root.testing(test_x)
        test_y_predicted_tree2=root2.testing(test_x)
        test_y_predicted_tree3=root3.testing(test_x)
        test_y_predicted_tree4=root4.testing(test_x)
        test_y_predicted=(test_y_predicted_tree1+test_y_predicted_tree2+test_y_predicted_tree3+test_y_predicted_tree4)/4
        predicted_list_y.append(test_y_predicted)
    
    
    predicted_list_y=np.array(predicted_list_y)
    
    
    
    
    
    
    for p in [1,3,5]:
        index=0
        sum_of_errors=0
        error=0
        
        limit=3
        
        
        new_feature_count_list=0.3
        
        
        
        
        
        for test_x in validation_data_X :
   
            test_y_original=validation_data_Y [index]  
            test_y_predicted=predicted_list_y[index]
    
    
            
    
            #for  tree in forest:#getting output from all trees in the forest     
                        #test_y_predicted=(test_y_predicted_tree3)/1 

        
        #predicted_list_y.append(test_y_predicted)
            error= get_error(test_y_original, test_y_predicted,p)
        #print(test_y_predicted)
    
    
    #mean_predicted_y=np.array(predicted_list_y).mean(0)
    
        
    #error_i= np.linalg.norm(test_y_original-mean_predicted_y)
    #sum_of_errors=sum_of_errors+error_i
            sum_of_errors= sum_of_errors+ error
            index=index+1
        
        
    
#avg_error=sum_of_errors/index
        avg_error=sum_of_errors/index   
    
        print('for validation  error with p= ',str(p),' k= ',str(k),'error = ',str(avg_error*100))
        #print(avg_error*100)

    
    
    
print('done')