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
