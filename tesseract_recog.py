from PIL import Image
import sys
import numpy as np
import pytesseract as pt
import cv2

class Process():
    def __init__(self):
        self.output = {'code': None, 'category': None}

        self.o = ('O', '0', 'D')
        self.b = ('B', '8', '6')

        self.category_dict = {}
        self.category_dict.update(dict.fromkeys(['我的設計'], 'Pattern'))
        self.category_dict.update(dict.fromkeys(['坦克背心'], 'Tank Top'))
        self.category_dict.update(dict.fromkeys(['短袖T恤'], 'Short Sleeve Tee'))
        self.category_dict.update(dict.fromkeys(['長袖襯衫'], 'Long Sleeve Dress Shirt'))
        self.category_dict.update(dict.fromkeys(['毛衣'], 'Sweater'))
        self.category_dict.update(dict.fromkeys(['連帽上衣'], 'Hoodie'))
        self.category_dict.update(dict.fromkeys(['大衣'], 'Coat'))
        self.category_dict.update(dict.fromkeys(['無袖連身裙'], 'Sleeveless Dress'))
        self.category_dict.update(dict.fromkeys(['短袖連身裙'], 'Short Sleeve Dress'))
        self.category_dict.update(dict.fromkeys(['長袖洋裝'], 'Long Sleeve Dress'))
        self.category_dict.update(dict.fromkeys(['圓形剪裁連身裙'], 'Round Dress'))
        self.category_dict.update(dict.fromkeys(['長袍'], 'Robe'))
        self.category_dict.update(dict.fromkeys(['氣球狀連身裙'], 'Balloon Hem Dress'))
        self.category_dict.update(dict.fromkeys(['遮陽帽'], 'Brimmed Hat'))
        self.category_dict.update(dict.fromkeys(['針織帽'], 'Knit Cap'))
        self.category_dict.update(dict.fromkeys(['鴨舌帽'], 'Brimmed Cap'))

    def load_image(self, img):
        img[:, :, 0] = 0
        self.img = self.rgb2gray(img)
        self.ori_h, self.ori_w = self.img.shape

    def split_str(self, img):
        img = 255 - img
        width = img.shape[1]
        flag, count = 0, 0
        char_img = []
        for i in range(width):
            detect = img[:,i].sum()
            if detect != 0:
                flag = 1
                count += 1
            if detect == 0 and flag == 1:
                flag = 0
                start = i - count
                end = start + count - 1
                char = img[:, start:end]
                c_w = char.shape[1]
                if c_w > 25:
                    sp = c_w // 2
                    c1 = char[:, :sp]
                    char_img.append(c1)
                    c2 = char[:, sp:]
                    char_img.append(c2)
                else:
                    char_img.append(char)
                count = 0
            if len(char_img) == 17:
                break
        return char_img

    def check_char(self, img):
        chars = self.split_str(img)
        for i, c in enumerate(self.code):
            # Check O, D, 0
            if c in self.o:
                ori_h, ori_w = chars[i].shape
                img = cv2.resize(chars[i],(int(43*ori_w/ori_h), 43))
                c_w = img.shape[1]
                if c_w < 15: 
                    self.reset_code(i, '0')    
                else:
                    if i == 1:
                        c_o = np.mean(img[:, :c_w//2])
                    else:
                        if np.mean(img[:, :c_w//2]) > c_o:
                            self.reset_code(i, 'D')
                        else:
                            self.reset_code(i, 'O')
            # Check 6, 8 ,B
            if c in self.b:
                ori_h, ori_w = chars[i].shape
                img = cv2.resize(chars[i],(int(43*ori_w/ori_h), 43))
                c_h, c_w = img.shape
                single = Image.fromarray(np.uint8(255-img))
                config = ('-l eng --oem 1 --psm 7')
                p_c = pt.image_to_string(single, config=config)
                if p_c in self.b:
                    self.reset_code(i, p_c)
                    continue
                else:
                    if c_w > 14:
                        self.reset_code(i, 'B')
                    else:
                        if abs(np.mean(img[:, :c_w//2]) - np.mean(img[:, c_w//2:])) < 5:
                            self.reset_code(i, '8')
                        else:
                            self.reset_code(i, '6')
        
    def reset_code(self, ind, char):
        temp = list(self.code)
        temp[ind] = char
        self.code = ''.join(temp)

    def rgb2gray(self, img):
        rgb_weights = [0.2989, 0.5870, 0.1140]
        img = np.dot(img, rgb_weights)
        return img

    def rm_char(self, text):
        char_list = text.split('-')
        sp = ('O0', '0O', 'D0', '0D', 'OD', 'DO')
        for i, l in enumerate(char_list):
            if len(l) > 4:
                for s in sp:
                    if s in l:
                        char_list[i] = l.replace(s, '0')
        text = '-'.join(char_list)
        return text

    def get_code(self):
        y1, y2 = 550/720, 593/720
        x = 920 / 1280
        img = self.img[int(y1*self.ori_h):int(y2*self.ori_h), int(x*self.ori_w):]
        ret, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
        whole = Image.fromarray(np.uint8(img))
        config = ('-l eng --oem 1 --psm 6')
        self.code = pt.image_to_string(whole, config=config)
        self.code = self.code.replace(' ', '')
        self.code = self.code.replace('.', '')
        self.code = self.code.upper()
        self.code = self.rm_char(self.code)
        self.code = self.code if len(self.code) == 17 else None
        if self.code is None:
            self.output['code'] = None
        else:
            self.check_char(img)
            self.output['code'] = self.code


    def get_category(self):
        y1, y2 = 510/720, 550/720
        x = 920 / 1280
        img = self.img[int(y1*self.ori_h):int(y2*self.ori_h), int(x*self.ori_w):]
        ret, img = cv2.threshold(img, 105, 255, cv2.THRESH_BINARY)
        img = Image.fromarray(np.uint8(img))
        config = ('-l chi_tra --oem 1 --psm 6')
        category = pt.image_to_string(img, config=config)
        category = self.check_category(category)
        self.output['category'] = self.category_dict[category] if category in self.category_dict else None

    def check_category(self, text):
        c_l = len(text)
        for c in text:
            if c in '連上' and c_l == 4:
                text = '連帽上衣'
            elif c in '氣球狀' and c_l == 6:
                text = '氣球狀連身裙'
            elif c in '我的設計' and c_l == 4:
                text = '我的設計'
            elif c in '圓形剪裁' and c_l == 7:
                text = '圓形剪裁連身裙'
            elif c in '針織' and c_l == 3:
                text = '針織帽'
            elif c in '長袍' and c_l == 2:
                text = '長袍'
            elif c in '鴨舌' and c_l == 3:
                text = '鴨舌帽'
            elif c in '遮陽' and c_l == 3:
                text = '遮陽帽'
            elif c in '大' and c_l == 2:
                text = '大衣'
            elif c in '坦克背心' and c_l == 4:
                text = '坦克背心'
            elif c in '毛' and c_l == 2:
                text = '毛衣'
            elif c in '無' and c_l == 5:
                text = '無袖連身裙'
            elif c in '洋裝' and c_l == 4:
                text = '長袖洋裝'
            elif c in '襯衫' and c_l == 4:
                text = '長袖襯衫'
            elif c in 'T恤' and c_l == 4:
                text = '短袖T恤'
        return text

if __name__ == '__main__':
    proc = Process()
    img = np.array(Image.open(sys.argv[1]), dtype=np.uint8)
    proc.load_image(img)
    proc.get_code()
    proc.get_category()
    print(proc.output)
