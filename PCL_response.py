import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def let2num(x):
    if x=='B': return 1
    elif x=='C': return 2
    elif x=='D': return 3
    elif x=='E': return 4


pcl = pd.read_csv('../Demographics_Survey/pcl_part3.csv')
pcl = pcl.groupby(['subject','DSM_block']).mean().reset_index()
pcl['blocknum'] = pcl.DSM_block.apply(let2num)

sns.set_style('ticks')
fig, ax = plt.subplots()
sns.lineplot(data=pcl,x='blocknum',y='score',units='subject',estimator=None,
            hue='subject',ax=ax)
ax.legend_.remove()
ax.set_xticks([1,2,3,4])
sns.despine(ax=ax)
ax.set_title('PCL block score by subject')
ax.set_xlabel('PCL block (B-E)')
