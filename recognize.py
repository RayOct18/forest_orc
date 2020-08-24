import cv2
import os
import numpy as np
import string
import pickle

class Recognize():
    def __init__(self):
        self.exception = ('Y-', 'YJ', 'TJ', 'T-', 'LY', 'PW', 'TW', 'YW', '4Y', '3T', 'DV', 'XC', 'DY', '-W', 'K3', 'VG')
        self.dic = self.make_dictionary()
        self.word = ('B', '8', '6', '0', 'O', 'D')

    def make_dictionary(self):
        dic = {}
        dic.update({'B': np.zeros((43,70), dtype=np.uint8)})
        dic.update({'8': np.zeros((43,70), dtype=np.uint8)})
        dic.update({'6': np.zeros((43,70), dtype=np.uint8)})
        dic.update({'0': np.zeros((43,70), dtype=np.uint8)})
        dic.update({'O': np.zeros((43,70), dtype=np.uint8)})
        dic.update({'D': np.zeros((43,70), dtype=np.uint8)})
        return dic

    def save_char(self, path, temp, total):
        name = os.path.split(path)[-1].split('.')[0]
        if total != 17:
            flag_i = 0
            for i in self.exception:
                if i in name:
                    flag_i = 1
                    ind = name.index(i)
                    #if self.dic[str(i)].sum() == 0 :
                    #    self.dic[str(i)] = temp.pop(ind)
                    temp.pop(ind)
                    name = name[:ind] + name[ind+2:]
            if flag_i == 0:
                #cv2.imshow('e', img)
                for j in range(len(temp)):
                    cv2.imshow('t', temp[j])
                    cv2.waitKey(0)
                cv2.waitKey(0)
            for i, c in enumerate(name):
                if c in self.word:
                    if self.dic[str(c)].sum() == 0:
                        self.dic[str(c)] = temp[i]
        else:
            for i, c in enumerate(name):
                if c in self.word:
                    if self.dic[str(c)].sum() == 0 :
                        self.dic[str(c)] = temp[i]

    def save_dictionary(self):
        f = open('dictionary.pkl', 'wb')
        pickle.dump(self.dic, f)
        f.close()


    def load_dictionary(self):
        self.dic = []
        with (open('dictionary.pkl', 'rb')) as fh:
            while True:
                try:
                    self.dic.append(pickle.load(fh))
                except EOFError:
                    break 
        self.key = list(self.dic[0].keys())
        self.char_array = np.array(list(self.dic[0].values())).astype('int8')

    def recognize(self, char):
        #cv2.imshow('a',self.char_array[12].astype('uint8'))
        #cv2.imshow('b',char)
        #cv2.waitKey(0)
        diff = abs(self.char_array - char.astype('int8'))
        mean_diff = np.mean(diff, axis=(1,2))
        min_diff = np.argmin(mean_diff)
        self.text = self.text +  str(self.key[min_diff])

    def process(self, path):
        img = cv2.imread(path)
        img[:, :, 2] = 0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ori_h, ori_w = img.shape
        y1, y2 = 550/720, 593/720
        x = 920 / 1280
        img = img[int(y1*ori_h):int(y2*ori_h), int(x*ori_w):]
        ori_h, ori_w = img.shape
        #img = cv2.resize(img, (int(25/ori_h * ori_w), 25))
        #img = cv2.GaussianBlur(img, (3, 3), 0)
        ret, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
        img = 255 - img
        #kernel = np.ones((3,3),np.uint8)
        #img = cv2.erode(img,kernel)
        #kernel = np.ones((2,2),np.uint8)
        #img = cv2.dilate(img,kernel,iterations = 1)

        width = img.shape[1]

        flag, count, total = 0, 0, 0
        self.text = ''
        temp = []
        for i in range(width):
            detect = img[:,i].sum()
            if detect != 0:
                flag = 1
                count += 1
            if detect == 0 and flag == 1:
                total += 1
                flag = 0
                start = i - count
                end = start + count - 1
                char = img[:, start:end]
                blank = np.zeros((43,70), np.uint8)
                c_h, c_w = char.shape
                m_h, m_w = 21-c_h//2, 35-c_w//2
                blank[m_h:m_h+c_h, m_w:m_w+c_w] = char
                temp.append(blank)
                count = 0
                #self.recognize(blank)
            if total == 17:
                break
        #gt = os.path.split(path)[-1].split('.')[0]
        #print(gt, self.text, gt == self.text)
        print(len(temp))
        self.save_char(path, temp, total)

if __name__ == '__main__':
    recog = Recognize()
    files = os.listdir('./figure/')
    #recog.load_dictionary()
    #files = ['MO-0B5C-5WD1-Q9D2.jpg']
    try:
        for f in files:
            path = os.path.join('./figure', f)
            recog.process(path)
        print(len(recog.dic))
    except:
        pass
    recog.save_dictionary()
