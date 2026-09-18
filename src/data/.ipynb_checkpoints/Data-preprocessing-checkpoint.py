#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from glob import glob


# ---------------------------
# ### Read the single CSV file
# ----------------------------

# In[10]:


single_filea_acc = pd.read_csv("../../data/raw/MetaMotion/MetaMotion/A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv")


# In[11]:


single_file_gyr = pd.read_csv("../../data/raw/MetaMotion/MetaMotion/A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv")


# In[12]:


# list all data in data/raw/metamotion

files = glob("../../data/raw/MetaMotion/MetaMotion/*.csv")
len(files)


# ### Extract features from file name.
# - These file names have something in common, as the data do not have the lable but file names have lable and we have to get it for updating data with lable.
# 

# In[41]:


data_path = ("../../data/raw/MetaMotion/MetaMotion/")
f = files[0]

participants = f.split("-")[0].replace(data_path,"")
label = f.split("-")[1]
category = f.split("-")[2].split("_")[0].rstrip("123")


# In[43]:


df = pd.read_csv(f)

df['participants'] = participants
df['label'] = label
df['category'] = category


# In[44]:


df


# # Read all file

# In[54]:


acc_df = pd.DataFrame()
gyr_df = pd.DataFrame()

acc_set = 1
gyr_set = 1

for f in files:
    participants = f.split("-")[0].replace(data_path,"")
    label = f.split("-")[1]
    category = f.split("-")[2].split("_")[0].rstrip("123")

    df = pd.read_csv(f)

    df['participants'] = participants
    df['label'] = label
    df['category'] = category

    if "Accelerometer" in f:
        df['set'] = acc_set
        acc_set += 1
        acc_df = pd.concat([acc_df, df])

    elif "Gyroscope" in f:
        df['set'] = gyr_set
        gyr_set += 1
        gyr_df = pd.concat([gyr_df, df])


# In[57]:


acc_df[acc_df['set'] == 10]


# ## Working with time

# In[68]:


acc_df.index = pd.to_datetime(acc_df['epoch (ms)'], unit="ms")
gyr_df.index = pd.to_datetime(gyr_df['epoch (ms)'], unit="ms")


# In[72]:


acc_df.drop(['epoch (ms)', 'time (01:00)','elapsed (s)'], axis=1, inplace= True)
acc_df


# In[73]:


gyr_df.drop(['epoch (ms)', 'time (01:00)','elapsed (s)'], axis=1, inplace= True)
gyr_df


# ## Create a function for all the above to wrap around

# In[78]:


files = glob("../../data/raw/MetaMotion/MetaMotion/*.csv")

def read_data_from_files(files):
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1
    data_path = ("../../data/raw/MetaMotion/MetaMotion/")
    f = files[0]
    for f in files:
        participants = f.split("-")[0].replace(data_path,"")
        label = f.split("-")[1]
        category = f.split("-")[2].split("_")[0].rstrip("123")

        df = pd.read_csv(f)

        df['participants'] = participants
        df['label'] = label
        df['category'] = category

        if "Accelerometer" in f:
            df['set'] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])

        elif "Gyroscope" in f:
            df['set'] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])

    acc_df.index = pd.to_datetime(acc_df['epoch (ms)'], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df['epoch (ms)'], unit="ms")

    acc_df.drop(['epoch (ms)', 'time (01:00)','elapsed (s)'], axis=1, inplace= True)
    gyr_df.drop(['epoch (ms)', 'time (01:00)','elapsed (s)'], axis=1, inplace= True)

    return acc_df, gyr_df


# In[80]:


acc_df, gyr_df = read_data_from_files(files)


# In[81]:


acc_df.head()


# In[94]:


## Merge the 2 data fram to one

data_merged = pd.concat([acc_df.iloc[:, :3],gyr_df], axis=1)
data_merged.columns = [
    "acc_x","acc_y","acc_z",
    "gyr_x","gyr_y", "gyr_z",
    "participants", "label", "Category", "set"]


# AS there are 2 sensors accelorometer is running 0.04 second and gyroscope on 0.08 seconds hence the diffrence in the measurements and the value they are providing
# 
# - Understanding hertz and changing the series of the frequency
# - refer offset data in pandas documents
# - it only works in timeseries data set hence we update the index with time stamp
# - use resmaple

# In[95]:


#we will be using the below function and applying it to entire df
data_merged[:1000].resample(rule="200ms").mean(numeric_only=True)


# In[99]:


sampling = {
    'acc_x':"mean", 'acc_y':"mean", 'acc_z':"mean", 'gyr_x':"mean", 'gyr_y':"mean", 'gyr_z':"mean", 
     'participants':"last", 'label':"last", 'Category':"last", 'set':"last"
}
data_merged[:1000].resample(rule="200ms").apply(sampling)


# In[102]:


# Spliting data on bases of day to reduce the unwanted data and save pc

days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]
data_resampled = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])
data_resampled['set'] = data_resampled['set'].astype("int")

data_resampled.info()


# In[103]:


data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")


# In[ ]:




