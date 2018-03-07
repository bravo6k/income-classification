import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
from itertools import chain
from sklearn.preprocessing import LabelEncoder
import itertools
import seaborn as sns
import squarify
import bar

def draw_num_hist(data,num_list):
    plt.subplots(figsize=(15,2))
    plt.suptitle('Histogram of Numerical Features')
    length=len(num_list)
    for i,j in itertools.zip_longest(num_list,range(length)):
        plt.subplot(1,6,j+1)
        plt.subplots_adjust(wspace=0.2,hspace=0.5)
        sns.distplot(data[i],bins=20)
        plt.title(i,size=15)
        plt.ylabel('')
        plt.xlabel('')
    plt.subplots_adjust(wspace=0.5,hspace=0.2,top=0.6)
    plt.show()

def draw_cate_bar(data,cate_list):
    plt.subplots(figsize=(15,8))
    plt.suptitle('Barplot of Categorical Features')
    length=len(cate_list)
    for i,j in itertools.zip_longest(cate_list,range(length)):
        plt.subplot(2,4,j+1)
        plt.subplots_adjust(wspace=0.2,hspace=0.5)
        sns.countplot(y=data[i],order=data[i].value_counts().index)
        plt.title(i,size=15)
        plt.ylabel('')
        plt.xlabel('')
    plt.subplots_adjust(wspace=0.5,hspace=0.2,top=0.9)
    plt.show()

def draw_cir2(data1,data2,x):

    class_dis1 = data1.groupby([x]).size().reset_index()
    class_dis1 = class_dis1.rename(index=str, columns={0:'per'})
    class_dis1 = class_dis1.set_index(x)

    class_dis2 = data2.groupby([x]).size().reset_index()
    class_dis2 = class_dis2.rename(index=str, columns={0:'per'})
    class_dis2 = class_dis2.set_index(x)

    f,(p1,p2) = plt.subplots(figsize=(15, 5), nrows=1, ncols=2)
    f.suptitle('Class Distribution')

    class_dis1['per'].plot.pie(autopct='%1.1f%%',colors=['g','lightblue','r'],wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' },ax=p1)
    class_dis2['per'].plot.pie(autopct='%1.1f%%',colors=['g','lightblue','r'],wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' },ax=p2)

    my_circle = plt.Circle( (0,0), 0.7, color='white')
#     p=plt.gcf()
#     p.gca().add_artist(my_circle)
    plt.subplots_adjust(wspace=0.5,hspace=0.2,top=0.8)
    p2.set_xlabel('Test Class')
    p1.set_xlabel('Train Class')
    p1.set_ylabel('')
    p2.set_ylabel('')
    plt.show()

def draw_bar_class1(data,x):
    nc_list = list((data["native-country"].value_counts() / data.shape[0])[0:10].index)
    if x!='native-country':
        plt.style.use('seaborn-white')

        p,(p1,p2) = plt.subplots(figsize=(10, 3), nrows=1, ncols=2)
        type_cluster = data.groupby([x,'class']).size().groupby(level=0).apply(lambda x:  x / float(x.sum()))
        type_cluster.unstack().plot(kind='bar',stacked=True, colormap= 'Pastel2',  grid=False,ax=p1)
        sns.countplot(y=data[x],order=data[x].value_counts().index,ax=p2)
        p2.set_ylabel('')
        plt.subplots_adjust(wspace=0.5,hspace=0.2,top=0.8)
        plt.show()
    else:
        plt.style.use('seaborn-white')
        d = data.copy()
        d = d.loc[d[x].isin(nc_list)]
        p,(p1,p2) = plt.subplots(figsize=(10, 3), nrows=1, ncols=2)
        type_cluster = d.groupby([x,'class']).size().groupby(level=0).apply(lambda x:  x / float(x.sum()))
        type_cluster.unstack().plot(kind='bar',stacked=True, colormap= 'Pastel2',  grid=False,ax=p1)
        sns.countplot(y=d[x],order=d[x].value_counts().index,ax=p2)
        p2.set_ylabel('')
        plt.subplots_adjust(wspace=0.5,hspace=0.2,top=0.8)
        plt.show()

def draw_bar_class2(data,x):
    nc_list = list((data["native-country"].value_counts() / data.shape[0])[0:10].index)
    if x!='native-country':
        plt.style.use('seaborn-white')
        work_class = pd.DataFrame(data[[x]].groupby([x]).size()).apply(lambda x:
                                                          100 * x / float(x.sum()))[0].reset_index()
        work_class = work_class.rename(index=str, columns={0:'per'})
        work_class = work_class.set_index(x)
        p,(p1,p2) = plt.subplots(figsize=(10, 3), nrows=1, ncols=2)
        type_cluster = data.groupby([x,'class']).size().groupby(level=0).apply(lambda x:  x / float(x.sum()))
        type_cluster.unstack().plot(kind='bar',stacked=True, colormap= 'Pastel2',  grid=False,ax=p1)
        work_class['per'].plot.pie(autopct='%1.1f%%',colormap= 'Pastel2',wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' },ax=p2)
        p2.set_ylabel('')
        plt.subplots_adjust(wspace=0.5,hspace=0.2,top=0.8)
        plt.show()
    else:
        plt.style.use('seaborn-white')
        d = data.copy()
        d = d.loc[d[x].isin(nc_list)]
        work_class = pd.DataFrame(d[[x]].groupby([x]).size()).apply(lambda x:
                                                          100 * x / float(x.sum()))[0].reset_index()
        work_class = work_class.rename(index=str, columns={0:'per'})
        work_class = work_class.set_index(x)
        p,(p1,p2) = plt.subplots(figsize=(10, 3), nrows=1, ncols=2)
        type_cluster = d.groupby([x,'class']).size().groupby(level=0).apply(lambda x:  x / float(x.sum()))
        type_cluster.unstack().plot(kind='bar',stacked=True, colormap= 'Pastel2',  grid=False,ax=p1)
        work_class['per'].plot.pie(autopct='%1.1f%%',colormap= 'Pastel2',wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' },ax=p2)
        p2.set_xlabel('')
        plt.subplots_adjust(wspace=0.5,hspace=0.2,top=0.8)
        plt.show()

def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))[0]

def number_encode_features(data):
    result = data.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders

def plot_confusion_matrix(x, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    x = x.astype('float') / x.sum(axis=1)[:, np.newaxis]
    plt.style.use('seaborn-white')
    plt.subplots(figsize=(8,4))
    plt.imshow(x, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = x.max() / 2.
    for i, j in itertools.product(range(x.shape[0]), range(x.shape[1])):
        plt.text(j, i, format(x[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if x[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def generate_interaction(data,feature_list):
    total = len(feature_list)*(len(feature_list)-1)/2
    step = 0
    for i,ai in enumerate(feature_list):
        for j,bj in enumerate(feature_list):
            if i<j:
                x = data[ai]
                y = data[bj]
                t = []
                for l in range(data.shape[0]):
                    t.append(str(x[l])+' '+ str(y[l]))
                data[ai+'_'+bj] = t
                step +=1
                bar.drawProgressBar(step/total)
