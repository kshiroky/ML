from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split as splt
import numpy as np
import pandas as pd
import cv2 as cv
import os

def extract_histogram(image):
    c = cv.imread(image)
    img = cv.calcHist([c], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    img_final = np.reshape(img, [512,1])
    cv.normalize(img_final, img_final)
    return img_final.flatten()

#enter your path to the directory
path = ('/home/nikolai/Downloads/train_task/train')

#to sort data (be carefull, check the output first)
raw_data =  os.listdir(path)
data = sorted(raw_data)

#to remove useless label
data.pop(0)

#to check output of function
c = data[1]
image = path + '/' + c
histogram = extract_histogram(image)
#print(histogram)

#to extract all images
im_final = {}
for each in data:
    c = path + '/' + each
    im = extract_histogram(c)
    im_final[each] = im

#to add label
cat_label = {}
dog_label = {}

dfc = pd.DataFrame(im_final)
dfc = dfc.transpose()

for each in data:
    if 'cat' in each:
        cat_label[each] = 0
    elif 'dog' in each:
        dog_label[each] = 1

cat_series = pd.Series(cat_label, name = 'label')
dog_series = pd.Series(dog_label, name = 'label')
label = cat_series.append(dog_series)

final_dfc = dfc.merge(label, right_index = True, left_index = True) 

a = final_dfc.iloc[512]

#to split the data
x_train, x_test, y_train, y_test=  splt(final_dfc[range(0,512)], final_dfc['label'],  test_size = 0.25, train_size = 0.75, random_state = 19)

#create the classifier
clf = LinearSVC(C = 0.6, random_state = 19)
train = clf.fit(x_train, y_train)

#to find theta
coef = train.coef_
coef = coef.flatten()
print(coef[10], coef[12], coef[317])

#this isn't necessary
score = train.score(x_test, y_test)
print('accuracy: ', score)

#to calculate mean F1-score
tp_0 = 0
tn_0 = 0
fp_0 = 0
fn_0 = 0

tp_1 = 0
tn_1 = 0
fp_1 = 0
fn_1 = 0

y_score = clf.predict(x_test)
y_test_ser = pd.DataFrame(y_test, x_test.index)
y_score_ser = pd.Series(y_score, x_test.index, name = 'scored_label')
y_for_test = np.array(y_test)
abc = y_test_ser.merge(y_score_ser, right_index = True, left_index = True)

for each in abc.index:
    if abc.label[each] == 0 and abc.scored_label[each] == 0:
        tp_0 += 1
        tn_1 += 1
    elif abc.label[each] == 0 and abc.scored_label[each] == 1:
        fn_0 += 1
        fp_1 += 1
    elif abc.label[each] == 1 and abc.scored_label[each] == 1:
        tn_0 += 1
        tp_1 += 1
    elif abc.label[each] == 1 and abc.scored_label[each] == 0:
        fp_0 += 1
        fn_1 += 1

#not necessary block
#print(tp_0,tn_0, fp_0, fn_0)
#print(tp_1,tn_1, fp_1, fn_1)

precision_0 = tp_0/(tp_0 + fp_0)
recall_0 = tp_0/(tp_0 + fn_0)
f1_0 = 2 * (precision_0*recall_0)/(precision_0+recall_0)

precision_1 = tp_1/(tp_1 + fp_1)
recall_1 = tp_1/(tp_1 + fn_1)
f1_1 = 2 * (precision_1*recall_1)/(precision_1+recall_1)

mean_f1 = (f1_0 + f1_1)/2
print(mean_f1)
    
#to classify test data
#enter your path to the directory
path_test = '/home/nikolai/Downloads/test'
raw_data_test = os.listdir(path_test)
data_test = sorted(raw_data_test)
data_test.pop(0)
data_test.pop(0)
#print(data_test)

i = 0
im_final_test = {}
to_predict_labels = {}


for each in data_test:
    c = path_test + '/' + each
    im = extract_histogram(c)
    im_final_test[each] = im

#filienames of images to label
im_test_1 = 'dog.1035.jpg'
im_test_2 = 'dog.1022.jpg'
im_test_3 = 'cat.1018.jpg'
im_test_4 = 'cat.1002.jpg' 
#to find the indexes of required images
for each in data_test:
    if  im_test_1 in each:
        to_predict_labels[each] = i
        i += 1
    elif im_test_2 in each:
        to_predict_labels[each] = i
        i += 1
    elif im_test_3 in each:
        to_predict_labels[each] = i
        i += 1
    elif im_test_4 in each:
        to_predict_labels[each] = i
        i += 1
    else:
        i += 1

test_im = pd.DataFrame(im_final_test)
test_im = test_im.transpose()

#add labels to test data
cat_label_test = {}
dog_label_test = {}

for each in data_test:
    if 'cat' in each:
        cat_label_test[each] = 0
    elif 'dog' in each:
        dog_label_test[each] = 1

cat_series_test = pd.Series(cat_label_test, name = 'label')
dog_series_test = pd.Series(dog_label_test, name = 'label')
label_test = cat_series_test.append(dog_series_test)

test_im_final = test_im.merge(label_test, right_index = True, left_index = True)
#dictionary with the indexes of required images
print(to_predict_labels)

#to score labels (change the number i in scr_data.iloc[i])
scr_data = test_im_final[range(0, 512)]
data_score_test = pd.DataFrame([scr_data.iloc[85],scr_data.iloc[22], scr_data.iloc[2],scr_data.iloc[18] ])
score_label = clf.predict(data_score_test)

#the result labels
print(score_label)