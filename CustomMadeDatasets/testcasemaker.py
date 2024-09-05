import pandas as pd
import numpy as nd
import torch
import os
# Custom Dataset 4 with torch.rand and 1-1000

c1 = torch.arange(1,1001).view(1000,1)
# print(c1.shape) #torch.Size([1000, 1])
c2 = torch.rand(1,1000).view(1000,1)
# print(c2.shape) # torch.Size([1000, 1])

# Combing them
c3 = torch.cat((c1,c2),dim=1)
# print(c3.shape) # torch.Size([1000, 2])
header = ['date','F1']
# Converting to csv
c3_np = c3.numpy()
c3_df = pd.DataFrame(c3_np)
c3_df.to_csv(os.path.join('test case','testcase4.csv'),index=False,header=header)

 # Custom Dataset 5 with torch.rand and the dates

 # I am going to use the pandas.date_range function for the dates
c4 = pd.date_range(start='1/1/2023', end='5/31/2023',freq='H') # This is going to do hourly for 151 days
# print(c4.size) #3601
#Convert from Time Index to dataframe
c4_df = pd.DataFrame(c4)

c5 = torch.rand(1,c4.size).view(c4.size,1)
# print(c5.shape) #torch.Size([3601, 1])
c5_np = c5.numpy()
c5_df = pd.DataFrame(c5_np)

c6=pd.concat([c4_df,c5_df],axis=1)
# print(c6.shape) #(3601, 2)
header = ['date','F1']
c6.to_csv(os.path.join('test case','testcase5.csv'),index=False,header = header)

import math
import torch
import pandas as pd
import os
import random

##Testing Basedin 5.1 Dataset
A1=random.randint(1,100)
A2=random.randint(1,100)

LA1 = 2*A2
LA2 = A2

HA1 = A1
HA2 = 2*A1

EA1 = A1
EA2 = A1

size=1000
####

test94x = torch.linspace(0,200*math.pi ,size).view(size,1)
# print(test94x.shape)

est94Ly=(LA1*torch.sin(1/2*test94x+math.pi)+LA2*torch.sin(test94x-math.pi)).view(size,1)
# print(test94Ly.shape)

# Combing them
test94L = torch.cat((test94x,test94Ly),dim=1)
# print(test94L.shape) # torch.Size([1000, 2])
header = ['date','F1']
# Converting to csv
test94L_np = test94L.numpy()
test94L_df = pd.DataFrame(test94L_np)
test94L_df.to_csv(os.path.join('test case','test94L.csv'),index=False,header=header)

test94Ey=(EA1*torch.sin(1/2*test94x+math.pi)+EA2*torch.sin(test94x-math.pi)).view(size,1)
# print(test94Ly.shape)

# Combing them
test94E = torch.cat((test94x,test94Ey),dim=1)
# print(test94L.shape) # torch.Size([1000, 2])
header = ['date','F1']
# Converting to csv
test94E_np = test94E.numpy()
test94E_df = pd.DataFrame(test94E_np)
test94E_df.to_csv(os.path.join('test case','test94E.csv'),index=False,header=header)

