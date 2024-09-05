import pandas as pd
import numpy as nd
import torch
import os

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
