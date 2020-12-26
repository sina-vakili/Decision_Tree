# --------- New Project


#Dataset libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from sklearn.tree import DecisionTreeRegressor

mat1 = loadmat('lable_17500.mat')
mat2 = loadmat('Mag_NOR_ARR_YES_PMU.mat')
mat3 = loadmat('Mag_VD_ARR_YES_PMU.mat')

My_Label = mat1['lable_17500']
Dataset1 = mat2['Mag_NOR_ARR_YES_PMU']
Dataset2 = mat3['Mag_VD_ARR_YES_PMU']
       
label = My_Label[0,:]
one_ = 0
zero_ = 0
my_gains = []

for one_zero in label:
    if one_zero:
        one_ +=1
    else:
        zero_ += 1
entropy_label = (-1*(zero_/(zero_+one_))*np.log2((zero_/(zero_+one_))) - (one_/(zero_+one_))*np.log2((one_/(zero_+one_))))
#print('entropy D /: ' , entropy_label)
#====================================================== Choose Row =============================================
for X in range(0,1615):
#for X in [0]:    
    calculate_list = []
    uniqe_detail = []
    #print ('You are inside the Row Loop number : ' , X )
    row = Dataset1[:,X]           #Chose your dataset here
    for x in range(0,17500):
        calculate_list.append((row[x] , label[x]))
                
#-----------------------------------------------------------> Entropy ------------------------------------------    
    uniqe_values = []
    uniqe = 1000
    #================================================== finding uniqe values=====================================
    calculate_list.sort()
    for values in range(0,17500):
        number_tuple = calculate_list[values]
        number = number_tuple[0]
        #print(number)
        
        if number != uniqe :
            uniqe_values.append(uniqe)
            uniqe = number
    
    uniqe_values.remove(1000)
    #print('my uniqe values are : ' , uniqe_values)
    
    #================================================= finding each uniqe 0 and 1 ===============================
    
    for each_uniqe in uniqe_values:
        #print('My Number is ' , each_uniqe , ' .')
        one = 0
        zero = 0
        
        for zero_one in calculate_list:
            if (each_uniqe == zero_one[0]):
                if ( zero_one[1] == 0 ) :
                    zero = zero + 1
                elif (zero_one[1] == 1 ):
                    one = one + 1
            detail = (each_uniqe , one , zero)
        uniqe_detail.append(detail)            
        #print('amount of zero in this uniqe number : ' , zero)
        #print('amount of one in this uniqe number : ' , one)
        
    #============================================== calculate Entropy =============================================
    project_entropy = []
    for uniqe_entropy in uniqe_detail:
        #print(uniqe_entropy)
        p = uniqe_entropy[1]
        n = uniqe_entropy[2]
        total = uniqe_entropy[1]+uniqe_entropy[2]
        if p==0 :
            each_entropy = 0 - ((n/total)*np.log2(n/total))
        elif n ==0 :
            each_entropy = ((-1*(p))/(total))*np.log2(p/total) 
        else:
            each_entropy = ((-1*(p))/(total))*np.log2(p/total) - ((n/total)*np.log2(n/total))
        project_entropy_detail = (uniqe_entropy[0] , p , total , each_entropy)
        project_entropy.append(project_entropy_detail)
       
               
    #============================================ Calculate I() ===================================================
    I_ROW = 0 
    for calculate_I in project_entropy:
        #print(calculate_I)
        I = (calculate_I[2]/17500) * calculate_I[3]
        I_ROW = I_ROW + I
    
    row_entropy = entropy_label - I_ROW
    
    print('----------------------------------------------------')
    print('I_ROW = ' , I_ROW)    
    print('Gain = Entropy(label) - I_ROW ' , row_entropy )
    print('----------------------------------------------------')
    #=========================================== Calculate Gain ===================================================
    
    row_gain = (I_ROW , row_entropy)
    my_gains.append(row_gain)
    #Gain = E(S) - I(R)
    
