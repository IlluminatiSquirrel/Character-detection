import numpy as np
import scipy
from scipy import io

#reading all .npy files
train_data1 = np.load("train1.npy")
train_data2 = np.load("train2.npy")
train_data3 = np.load("train3.npy")
train_data4 = np.load("train4.npy")
test_data1 = np.load("test1.npy")
test_data2 = np.load("test2.npy")
test_data1_1 = np.load("test1.1.npy")
test_data1_2 = np.load("test1.2.npy")
test_data1_3 = np.load("test1.3.npy")
test_data1_4 = np.load("test1.4.npy")
test_data2_1 = np.load("test2.1.npy")
test_data2_2 = np.load("test2.2.npy")
test_data2_3 = np.load("test2.3.npy")
test_data2_4 = np.load("test2.4.npy")

#reading and seperating all .dat files into labels, boundary boxes and word ends
train_labels1 = np.loadtxt("train1.dat", dtype = 'str', usecols = [0])
train_bounding_boxes1 = np.loadtxt("train1.dat", dtype = int, usecols = [1,2,3,4])
train_labels2 = np.loadtxt("train2.dat", dtype = 'str', usecols = [0])
train_bounding_boxes2 = np.loadtxt("train2.dat", dtype = int, usecols = [1,2,3,4])
train_labels3 = np.loadtxt("train3.dat", dtype = 'str', usecols = [0])
train_bounding_boxes3 = np.loadtxt("train3.dat", dtype = int, usecols = [1,2,3,4])
train_labels4 = np.loadtxt("train4.dat", dtype = 'str', usecols = [0])
train_bounding_boxes4 = np.loadtxt("train4.dat", dtype = int, usecols = [1,2,3,4])
test_labels1 = np.loadtxt("test1.dat", dtype = 'str', usecols = [0])
test_bounding_boxes1 = np.loadtxt("test1.dat", dtype = int, usecols = [1,2,3,4])
test_word_ends1 = np.loadtxt("test1.dat", dtype = int, usecols = [5])
test_labels2 = np.loadtxt("test2.dat", dtype = 'str', usecols = [0])
test_bounding_boxes2 = np.loadtxt("test2.dat", dtype = int, usecols = [1,2,3,4])
test_word_ends2 = np.loadtxt("test2.dat", dtype = int, usecols = [5])

#given a page and bounding box coordinates finds and obtains all letters from that page
def get_letters (data, bounding_boxes):
    x=[]
    for i in xrange(bounding_boxes.shape[0]):
        left = bounding_boxes[i,0]
        right = bounding_boxes[i,2]
        top = data.shape[0] - bounding_boxes[i,3]
        bottom = data.shape[0] - bounding_boxes[i,1]
        letter = data[top:bottom,left:right]
        x.append(letter)
    return x

#adds all obtained letters from training pages into one list
def train_feature_vectors():
    features1 = get_letters(train_data1, train_bounding_boxes1)
    features2 = get_letters(train_data2, train_bounding_boxes2)
    features3 = get_letters(train_data3, train_bounding_boxes3)
    features4 = get_letters(train_data4, train_bounding_boxes4)
    return features1+features2+features3+features4

#finds the biggest heigth and biggest width from given letter set
def biggest_dimension(train,test):
    all_letters = train+test
    biggest_height = 0
    biggest_width = 0
    for i in xrange(len(all_letters)):
        letter_height = all_letters[i].shape[0]
        letter_width = all_letters[i].shape[1]
        if (letter_height > biggest_height):
            biggest_height = letter_height
        if (letter_width > biggest_width):
            biggest_width = letter_width
    return biggest_height, biggest_width

#padds all all letters from a given letter set
def letter_padding(data,biggest_height,biggest_width):
    new_vectors = []
    for i in xrange(len(data)):
        letter_height = data[i].shape[0]
        letter_width = data[i].shape[1]
        if (letter_height==biggest_height) and (letter_width==biggest_width): #if letter already has
            new_vectors.append(data[i])                                       #biggest height and biggest width 
        else:                                                                 #there is no need to pad it
            height_diff = biggest_height - letter_height                          
            width_diff = biggest_width - letter_width
            if (height_diff % 2 != 0):                           #if difference is odd number it padds
                top_pad = height_diff / 2                        #one side 1 pixel more than the other side
                bottom_pad = height_diff / 2 + 1
            else:
                top_pad = height_diff / 2
                bottom_pad = height_diff / 2
            if (width_diff % 2 != 0):
                left_pad = width_diff / 2
                right_pad = width_diff / 2 + 1
            else:
                left_pad = width_diff / 2
                right_pad = width_diff / 2
        padded_letter = np.pad(data[i], ((top_pad,bottom_pad), (left_pad,right_pad)), mode='constant', constant_values=255)
        new_vectors.append(np.hstack(padded_letter))
    return new_vectors

#returns train and test data as 2d matrices with equal size vectors
def get_data(train,test):
    biggest_height, biggest_width = biggest_dimension(train,test)
    padded_train = letter_padding(train,biggest_height,biggest_width)
    padded_test = letter_padding(test,biggest_height,biggest_width)
    matrix_rows_train = len(padded_train)
    matrix_rows_test = len(padded_test)
    matrix_columns = biggest_height*biggest_width
    train_data = np.reshape(padded_train,[matrix_rows_train,matrix_columns])
    test_data = np.reshape(padded_test,[matrix_rows_test,matrix_columns])
    return train_data,test_data

#calculates principal components and transform the data using them (code from the lab class)
def pca (train,test):
    covx = np.cov(train, rowvar=0)
    N = covx.shape[0]
    w, v = scipy.linalg.eigh(covx, eigvals=(N-40, N-1))
    v = np.fliplr(v)
    pcatrain_data = np.dot((train - np.mean(train)), v)
    pcatest_data = np.dot((test- np.mean(test)), v)
    return pcatrain_data, pcatest_data

#classifies test data letters (code from the lab class)
def classify(train, train_labels, test, test_labels, features=None):
    # Use all feature is no feature parameter has been supplied
    if features is None:
        features=np.arange(0, train.shape[1])
    # Select the desired features from the training and test data
    train = train[:, features]
    test = test[:, features]
    # Super compact implementation of nearest neighbour 
    x= np.dot(test, train.transpose())
    modtest=np.sqrt(np.sum(test*test,axis=1))
    modtrain=np.sqrt(np.sum(train*train,axis=1))
    dist = x/np.outer(modtest, modtrain.transpose()) 
    nearest=np.argmax(dist, axis=1)
    mdist=np.max(dist, axis=1)
    label = train_labels[nearest]
    score = (100.0 * sum(test_labels==label))/len(label)
    return score, label

#prints classifier's output for each character as a string with spaces between words
def print_text (labels, word_ends):
    text = ""
    for i in xrange(len(labels)):
        text = text + labels[i]
        if (word_ends[i]==1):
            text = text + " "    
    return text

#function to compute the score and labels without using dimensiolity reduction
def no_pca_test(test,test_labels,word_ends):
    train_data,test_data = get_data(train,test)
    score,label = classify(train_data-np.mean(train_data), train_labels, test_data-np.mean(test_data), test_labels)
    text = print_text (label, word_ends)
    return str(score)+"\n"+text+"\n"

#function to compute the score and labels using dimensiolity reduction
def pca_test(test,test_labels,word_ends):
    train_data,test_data = get_data(train,test)
    pcatrain_data, pcatest_data = pca(train_data,test_data)
    score,label = classify(pcatrain_data, train_labels, pcatest_data, test_labels, xrange(1,11))
    text = print_text (label, word_ends)
    return str(score)+"\n"+text+"\n"

#runs and outputs the results of all tests
def run_all_tests():
    print "Trial 1" + "\n"
    print "Test1: " + no_pca_test(test1,test_labels1,test_word_ends1)
    print "Test2: " + no_pca_test(test2,test_labels2,test_word_ends2)
    print "Trial 2" + "\n"
    print "Test1_1: " + no_pca_test(test1_1,test_labels1,test_word_ends1)
    print "Test 1_2: " + no_pca_test(test1_2,test_labels1,test_word_ends1)
    print "Test 1_3: " + no_pca_test(test1_3,test_labels1,test_word_ends1)
    print "Test 1_4: " + no_pca_test(test1_4,test_labels1,test_word_ends1)
    print "Test 2_1: " + no_pca_test(test2_1,test_labels2,test_word_ends2)
    print "Test 2_2: " + no_pca_test(test2_2,test_labels2,test_word_ends2)
    print "Test 2_3: " + no_pca_test(test2_3,test_labels2,test_word_ends2)
    print "Test 2_4: " + no_pca_test(test2_4,test_labels2,test_word_ends2)
    print "Trial 3" + "\n"
    print "Test 1: " + pca_test(test1,test_labels1,test_word_ends1)
    print "Test 1_1: " + pca_test(test1_1,test_labels1,test_word_ends1)
    print "Test 1_2: " + pca_test(test1_2,test_labels1,test_word_ends1)
    print "Test 1_3: " + pca_test(test1_3,test_labels1,test_word_ends1)
    print "Test 1_4: " + pca_test(test1_4,test_labels1,test_word_ends1)
    print "Test 2: " + pca_test(test2,test_labels2,test_word_ends2)
    print "Test 2_1: " + pca_test(test2_1,test_labels2,test_word_ends2)
    print "Test 2_2: " + pca_test(test2_2,test_labels2,test_word_ends2)
    print "Test 2_3: " + pca_test(test2_3,test_labels2,test_word_ends2)
    print "Test 2_4: " + pca_test(test2_4,test_labels2,test_word_ends2)

train = train_feature_vectors()

test1 = get_letters(test_data1, test_bounding_boxes1)
test1_1 = get_letters(test_data1_1, test_bounding_boxes1)
test1_2 = get_letters(test_data1_2, test_bounding_boxes1)
test1_3 = get_letters(test_data1_3, test_bounding_boxes1)
test1_4 = get_letters(test_data1_4, test_bounding_boxes1)

test2 = get_letters(test_data2, test_bounding_boxes2)
test2_1 = get_letters(test_data2_1, test_bounding_boxes2)
test2_2 = get_letters(test_data2_2, test_bounding_boxes2)
test2_3 = get_letters(test_data2_3, test_bounding_boxes2)
test2_4 = get_letters(test_data2_4, test_bounding_boxes2)

train_labels = np.hstack((train_labels1, train_labels2, train_labels3, train_labels4))

run_all_tests()