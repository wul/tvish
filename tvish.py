#!/usr/bin/env python3
"""
Please see LICENSE for copyright
Author: 'Wu Li' <li.koun@gmail.com>

上海交通违规查询
========================================

This is a command line tool that checks if user violated the traffic security code

"""

import sys
import os
from PIL import Image
import glob
import numpy as np
from logistic import *
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import configparser
import requests
import re

#This is url for search the record of violating traffic safe code
URL='http://www.shjtaq.com/Server4/dzjc_new.asp?'
IDENTIFY_URL='http://www.shjtaq.com/Server4/validatecode.asp'


def img2vector(im):
    '''
    Turn image object into grayed image, then get all points and distinguish the point 
    by 1 or 0 according to their gray value
    '''

    gi = im.convert('L')
    lst = []
    for point in gi.getdata():
        if point > 130:
            lst.append(1)
        else:
            lst.append(0)

    return lst

def loadDataSet():
    '''
    Loads the training examples
    '''

    matrix = np.loadtxt('examples/dataSet.txt')
    X = matrix[:, 0:-1]
    y = matrix[:,-1]

    return (X,y)


def cropImage(pathname):
    '''
    Crops the downloaded image. The downloaded image contains 4 digital numbers and have
    a geometry 10x40, each number in the image is 10x10
    '''
    im = Image.open(pathname)

    im1 = im.crop((0,0,10,10))
    im2 = im.crop((10,0,20,10))
    im3 = im.crop((20,0,30,10))
    im4 = im.crop((30,0,40,10))
        
    return (im1,im2,im3,im4)


def loadTest(pathname):
    '''
    Loads the test data. It loads the image file, then turn it into numeric vector
    '''
    
    lsts = []
    for im in cropImage(pathname):
        lsts.append(img2vector(im))

    X = np.array(lsts)
    return X

def predictViaLR(X, y, X_test):
    '''
    Predicts the numbers via logistic regression algorithm
    '''

    print("Calculate cost with all zeros theta and lambda=1")
    theta = np.zeros((len(y)+1, 1))
    lmd = 1
    allTheta = oneVsAll(np.hstack((np.ones((m, 1)), X)), y, lmd, 10)

    values = predict (np.hstack((np.ones((X_test.shape[0],1)),X_test)), allTheta)
    return values


def numbers2str(values):
    '''
    Turns number list into string
    '''
    return ''.join([str(int(x)) for x in values])

def printNumbers(values):
    '''
    Prints the number list
    '''
    print("The numbers are:%s" % numbers2str(values))

def predictViaSVM(X, y, X_test):
    '''
    Uses the SVM in sklearn package to predict the numbers. Like logistic regression, it works
    well.
    '''
    model = LinearSVC()
    model.fit(X, y)
    values = model.predict(X_test)
    return values

def predictViaKNN(X, y, X_test):
    '''
    kNN is a bad choice to predict the numbers, it can not distingusih some number well. For example,
    8 and 0
    '''
    knn = KNeighborsClassifier()
    knn.fit(X, y)
    values = knn.predict(X_test)
    return values


    
def predictViaDT(X,y,X_test):
    '''
    Predicts the numbers via decision tree
    '''
    clf = tree.DecisionTreeClassifier()
    clf.fit(X,y)
    values = clf.predict(X_test)
    return values
    
def collectData(pathname):
    '''
    Convert all sample images to training examples for later use
    '''
    files = glob.glob("%s/*.bmp" % pathname)
    lsts = []
    for file in files:
        filename = os.path.basename(file)
        numbers = filename.split(".")[0] 

        ims = cropImage(file)
        
        idx = 0
        for i in ims:
            lst = img2vector(i)
            lst.append(int(numbers[idx]))
            idx += 1
            lsts.append(lst)

    print(lsts)
    np.savetxt('examples/dataSet.txt', np.array(lsts))

        
def getIdentifyingCode(sess):
    '''
    Gets the image and saves cookies into session
    '''

    r = sess.get(IDENTIFY_URL, params={'m':np.random.random()})
    pathname = "/tmp/code.bmp"
    with open(pathname, 'wb') as outf:
        outf.write(r.content)
    
    return pathname

def displayResult(content):
    pattern = re.compile('(您查询的车牌号为.*的小型汽车,目前在本市没有未处理的交通违法记录！)')
    for line in content.splitlines():
        m = pattern.search(line)
        if m:
            print(m.group(1))
            return 
    print("你有交通违法记录，请登录%s进行查询"%URL)
    m = re.search(r"车牌号(.*)状态", content)
  
    print(m.group(1))

if __name__ == '__main__':

    if len(sys.argv) > 1: 
        print("Building training examples....")
        collectData("./examples")
        print("Done!")
        sys.exit(0)
    

    config = configparser.ConfigParser()
    config.read([os.path.expanduser('~/.tvish')])

    
    
    #Loads the training examples
    X, y = loadDataSet()
    m, n = X.shape
    sess = requests.Session()

    #Get test images from website
    for section in config.sections():

        pathname = getIdentifyingCode(sess)
        X_test = loadTest(pathname)


        #
        # Logistic regression, SVM and DT are all works fine for this site
        #

        #Using logistic regression
        values = predictViaLR(X,y,X_test)
        printNumbers(values)

        #Using SVM
        values = predictViaSVM(X,y,X_test)
        printNumbers(values)
        
        #Decision Tree
        values = predictViaDT(X, y, X_test)
        printNumbers(values)
    
        s = numbers2str(values)


        #try sumbit form
        licenseNumber = section

        province = licenseNumber[0]
        encoding='GB2312'
        data = {'type1': config[licenseNumber]['Type'].encode(encoding),
                'cardqz': province.encode(encoding),
                'carnumber': licenseNumber[1:],
                'fdjh': config[licenseNumber]['EngineNumber'],
                'verify': s,
                'WrongCode':'',
                'act':'search',
                'submit': u' 提 交 '.encode(encoding),
            }

        needle=u'本市查询结果'



        r = sess.post(URL, data=data)

        # this site is still using GB2312 for character encoding
        content = r.content.decode('gbk')
        
        displayResult(content)
