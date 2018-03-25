path='/users/chithkala/data.txt'
import numpy as np
import math
from numpy import matrix
import pandas as pd
index=[]
def main():
   data=[]
   data=np.loadtxt(path,delimiter=',') #load the file data, Rename B to data
   #print(data)

   m,n=data.shape                      #get no. of data and attributes
   target=data[:,n-1]
      # Since it is boolean, sum will give us no of positives
  # print('target=',target)
   pos=np.sum(target)               
   neg=m-pos
   print('positive=',pos)
   print('negative=',neg) 
   if(pos==m):
       print('Decision:YES')
   #else decision is no
   elif(neg==m):
       print('Decision:NO')                 
   #calculate the entropy hence call the function
   E=entropy(pos,m)
   print('Entropy=',E)
   make_node(data,E,n,m)                    

def make_node(data,E,n,m):    
   # Distinct attribute postive count
   attr_pos_count=[]
   # G is gain for each attribute.
   G=[]
   counts=0
   m,n=data.shape
   # Target is the last column of the data.
   target=data[:,n-1]               #get the target attribute
   #print('target',target)
   #print('data=',data)
   k=0
   for j in range (1,n-1):          #check for every attributes
       col=data[:,j]
       #print('col=',col)
       temp=[]
  	# For every attr column we are counting the unique values & counts
       unique,counts = np.unique(col,return_counts=True)
       #print('uniq and cou=',unique,counts)
       # Replace this with unique
       b=int(max(col))              
       #print (s)
       for i in unique:      #check every value in that attribute
           s=0
           for c in range (0,m):    #for every days
               if (col[c]==i and target[c]==1): 
                   s=s+1            #get all the yes sum
           #print('count=',s)
           k=k+1
           temp.insert(k,s)         #and store it in the list
           #print(temp)
       attr_pos_count.insert(j-1,temp)
       #print(attr_pos_count)
       r=gain(E,counts,attr_pos_count,m,j,b)   
	#calculate the gain by calling the function
       #print ('gain=',r)
       # print('j=',j)
       G.insert(j-1,r)
   print('Gain=',G)
   indx=G.index(max(G)) + 1
   index.insert(j-1,indx)
   print('list index:',index)
   build_tree(data,indx,m,n)


def gain(E,counts,l,m,j,b):
   #print(E)
   ent=0.0
   e=[]
   a=0.0
   for i in range(0,len(counts)):     #for every value in that attribute
       #print(counts,b,j,m)
       c=entropy(l[j-1][i],counts[i])
       #print(counts[i]/m) 
       a=a + (counts[i]/m) * entropy(l[j-1][i],counts[i]) #calculate the entropy of the values of the attribute
   #print ('a=',a)
   #calculate the gain 
   G=E-a                              
   #print ('Gain=',G,j)
   return G                        #return the gain

def entropy(pos,m):
   prob= pos/m                   #get the proportion for the positive values
   prob2=1-prob                  #get the proportion for negative values
   if (prob==1 or prob2==1):                 #if proportion comes as 1
       E=0                       #means entropy is 0
   else:
       E= -prob*math.log(prob,2) - prob2*math.log(prob2,2) #else calculate the entropy
   #print (prob,prob2)
   #print ('Entropy is=',E)
   return E                      #return the entropy value

def build_tree(data,indx,m,n):
   # get the unique value for that root attribute 
   uniq_val,uniq_counts=np.unique(data[:,indx],return_counts=True)
   #print(uniq_val,uniq_counts)
   #sort the value for the root attribute according to the values of the attribute
   sorted_val=data[data[:,indx].argsort()]
   target_val=sorted_val[:,n-1]
   #print(sorted_val)
   #print(target_val)
   attr_val=[]
   uc =  uniq_counts.tolist()
   uc.insert(0,0)
   x=0
   #print('sort=\n',sorted_val)
   #As we get the root attribute we will remove that value
   attr_remv=np.delete(sorted_val,indx,1)
   #print('remv=',attr_remv)
   #print('indx before=',indx)
   len_uval = len(uniq_val)
   for k in range(0,len_uval):
       #print('indx after=',indx)
       x=x+uc[k]
       #divide the root attribute according to the different values it have
       attr_val.append(attr_remv[x:x+uc[k+1],:])
       #print('attr_val=\n',attr_val[k])
       post=0
       #check for every value of the root attribute and count the yes and no to get the entropy
       for l in range (0,m):
           if(sorted_val[:,indx][l]==k and target_val[l]==1):
               post=post+1
       neg=uniq_counts[k]-post
       #sorted_remv=np.delete(sorted_val,indx,1)
       #print(sorted_remv)
       print('attr_value=',k,' postive=',post)
       print('attr_value=',k,' negative=',neg)
       #if the yes count equals the no. of count of that value then decision is yes
       if(post==uniq_counts[k]):
           print('Decision:YES')
           continue
       #else decision is no
       elif(neg==uniq_counts[k]):
           print('Decision:NO')
           continue
       attr_count=uniq_counts[k]
       #print(attr_count)
       #calculate entropy
       E1=entropy(post,attr_count)
       print('Entropy for the value of root attribute=',E1)               
       #attr_remv=np.delete(attr_val[k],indx,1)
       #send all data recursively
       make_node(attr_val[k],E1,n,attr_count)
main()
