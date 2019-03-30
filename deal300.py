import os
import glob
import cv2
import numpy as np
import copy
import random
import shutil
from scipy import io
def toCropRandom(img, midbox ,side_range,bbox):
    # should keep bbox in target box
    inbox=[0,0,-1,-1]
    tms = 0
    side = 0
    while (inbox[2] < inbox[0] or inbox[3] < inbox[1]) and tms < 3:
        tms = tms + 1
        side_st = int(max([side_range[0], bbox[2] - bbox[0] + 2, bbox[3] - bbox[1] + 1]))
        if side_st >= int(side_range[1]):
            break
        side = random.randint(side_st, int(side_range[1]))
        st = [bbox[2] - side + side/2.0, bbox[3]-side+side/2.0, bbox[0]+side/2.0, bbox[1]+side/2.0]
        inbox = [max(st[0],midbox[0]), max(st[1],midbox[1]), min(st[2],midbox[2]), min(st[3], midbox[3])]
    # print 'haha'
    if inbox[2] < inbox[0] or inbox[3] < inbox[1]:
        return -1,-1,-1
    # print 'ha1'
    center = (random.randint(int(inbox[0]),int(inbox[2])), random.randint(int(inbox[1]), int(inbox[3])))

    x = max(0,int(center[0] - side/2.0))
    y = max(0,int(center[1] - side/2.0))
    if x + side > img.shape[1]:
        x = img.shape[1] - side
    if y + side > img.shape[0]:
        y = img.shape[0] - side
    # print 'ok'

    # cv2.rectangle(temp,(int(midbox[0]), int(midbox[1])),(int(midbox[2]),int(midbox[3])),128)
    # cv2.rectangle(temp,(int(st[0]), int(st[1])),(int(st[2]),int(st[3])),255)
    # cv2.rectangle(temp, (int(inbox[0]), int(inbox[1])), (int(inbox[2]), int(inbox[3])), 10)
    #
    # cv2.rectangle(temp,(x,y),(x+side,y+side),50)
    # cv2.imshow('temp',temp)

    return x,y,side

def getTagStr(tag):
    str1 = ''
    for i in range(0,tag.shape[0]):
        str1 = str1+' ' + str(tag[i][0]) +' ' + str(tag[i][1])
    return str1

root_path = '/data2/interns/ykwang/git/256_1k_train_data/'
save_root = root_path + 'imgs/'
if os.path.exists(save_root):
    shutil.rmtree(save_root)
os.makedirs(save_root)
fw = open(root_path + 'data_tag.list','w')

ans = 0

imgs = glob.glob('/data5/public_datasets/face/alignment/300w/*/*.png')
imgs = glob.glob('/data5/public_datasets/face/alignment/helen/testset/*.jpg')
for fg in imgs:
#    break
    if ans >= 1000:
        break
#    ans += 1
    names = fg.split('/')
    fidname = names[-1][:-4]
    img = cv2.imread(fg, -1)
    ftag = open(fg[:-3]+'pts').readlines()
    dots = []
    print(fg)
    for i in range(3,3+68):
        sl = ftag[i].strip('\n').split(' ')
        dot = [float(sl[0]) , float(sl[1])]
        # cv2.rectangle(img,(int(dot[0]),int(dot[1])),(int(dot[0]+1),int(dot[1]+1)),(0,255,0))
        dots.append(dot)
    ptd2 = np.array(dots)
    ptd2 = np.reshape(ptd2,(-1,2))

    if len(img.shape) == 2:
        gray = copy.deepcopy(img)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#        gray = img   
    for k in range(0,1):
        ndots = copy.deepcopy(ptd2)
        bbox = [np.min(ndots[:, 0]) - 2, np.min(ndots[:, 1]) - 2, np.max(ndots[:, 0]) + 2, np.max(ndots[:, 1]) + 2]

        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > img.shape[1] or bbox[3] > img.shape[0]:
            continue

        wbox = [np.min(ndots[18:, 0]), np.min(ndots[18:, 1]), np.max(ndots[18:, 0]), np.max(ndots[18:, 1])]
        nsize = max(wbox[2] - wbox[0], wbox[3] - wbox[1])
        side_range = [min([nsize * 1.6, img.shape[0], img.shape[1]]), min([nsize * 2.6, img.shape[0], img.shape[1]])]
        wmid = [(wbox[0] + wbox[2]) / 2.0, (wbox[1] + wbox[3]) / 2.0]
        midbox = [wmid[0] - nsize / 4.0, wmid[1] - nsize / 4.0, wmid[0] + nsize / 4.0, wmid[1] + nsize / 4.0]

        x, y, side = toCropRandom(gray, midbox, side_range, bbox)

        if x < 0 or y < 0:
            continue
        crop = gray[y:y + side, x:x + side]
        ndots[:, 0] -= x
        ndots[:, 1] -= y

        crop = cv2.resize(crop, (256, 256))
        scale = 256.0 / side
        ndots *= scale 
        fname = save_root +fidname + '_' + str(k) + '_' + str(ans) + '.bmp'
        ans = ans + 1
        print(fname)
        cv2.imwrite(fname, crop)
        fw.write(fname+getTagStr(ndots)+'\n')
#    break

file_path = ['AFW', 'HELEN', 'IBUG', 'LFPW']
file_path = ['LFPW']

for f in file_path:
    break
#    ans += 1
#    if ans >= 10000:
#        break
    imgs = glob.glob('/data5/public_datasets/face/alignment/300W_LP/' + f + '/*.jpg')
    print(f)
    for fg in imgs:
        if ans >= 10000:
            break
        print(fg)
        names = fg.split('/')
        fidname = names[-1][:-4]
        img = cv2.imread(fg, -1)
        ftag = io.loadmat(fg[:-3]+'mat')

        ptd2 = ftag['pt2d']

        ptd2 = np.transpose(ptd2)
        ptd2 = np.reshape(ptd2,(-1,2))
        dots = []
        for i in ptd2:
#            sl = ftag[i].strip('\n').split(' ')
            dot = i
            # cv2.rectangle(img,(int(dot[0]),int(dot[1])),(int(dot[0]+1),int(dot[1]+1)),(0,255,0))
            dots.append(dot)
        ptd2 = np.array(dots)
        ptd2 = np.reshape(ptd2,(-1,2))

        if len(img.shape) == 2:
            gray = copy.deepcopy(img)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for k in range(0,1):
            ndots = copy.deepcopy(ptd2)
            bbox = [np.min(ndots[:, 0]) - 2, np.min(ndots[:, 1]) - 2, np.max(ndots[:, 0]) + 2, np.max(ndots[:, 1]) + 2]

            if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > img.shape[1] or bbox[3] > img.shape[0]:
                continue

            wbox = [np.min(ndots[18:, 0]), np.min(ndots[18:, 1]), np.max(ndots[18:, 0]), np.max(ndots[18:, 1])]
            nsize = max(wbox[2] - wbox[0], wbox[3] - wbox[1])
            side_range = [min([nsize * 1.6, img.shape[0], img.shape[1]]), min([nsize * 2.6, img.shape[0], img.shape[1]])]
            wmid = [(wbox[0] + wbox[2]) / 2.0, (wbox[1] + wbox[3]) / 2.0]
            midbox = [wmid[0] - nsize / 4.0, wmid[1] - nsize / 4.0, wmid[0] + nsize / 4.0, wmid[1] + nsize / 4.0]

            x, y, side = toCropRandom(gray, midbox, side_range, bbox)

            if x < 0 or y < 0:
                continue
            crop = gray[y:y + side, x:x + side]
            ndots[:, 0] -= x
            ndots[:, 1] -= y

            crop = cv2.resize(crop, (256, 256))
            scale = 256.0 / side
            ndots *= scale
            fname = save_root +  fidname  + '_'  + str(k)+ '_' + str(ans) + '.bmp'
            print(fname)
#            fname = img_path +fidname+'_'+str(k)+ '_' + str(ans) + '.bmp'
            #if fname == '/data2/interns/ykwang/pytorch_project/data_test/imgs/AFW_2751381965_1_1_0_94.bmp' or fname == '/data2/interns/ykwang/pytorch_project/data_test/imgs/AFW_3284354538_3_3_0_75.bmp' or fidname == '/data2/interns/ykwang/pytorch_project/data_test/imgs/AFW_4022732812_2_0_0_8.bmp':
            cv2.imwrite(fname, crop)
            fw.write(fname+getTagStr(ndots)+'\n')
            ans += 1

imgs = glob.glob('/data5/public_datasets/face/alignment/300VW_Dataset_2015_12_14/*/frames/*.jpg')
for fg in imgs:
    break
#    print(ans)
#    if ans >= 100:
#        break
    names = fg.split('/')
    fidname = names[-1][:-4]
    img = cv2.imread(fg, -1)
    ftag_path = fg[0:-17] + 'annot/' + fidname + '.pts'
    nums = fg[-21:-18]
    print('----------------------')
    print(nums)
    print(fg)
    print(ftag_path)
    ftag = open(ftag_path).readlines()
    dots = []
    for i in range(3,3+68):
        sl = ftag[i].strip('\n').split(' ')
        dot = [float(sl[0]) , float(sl[1])]
        # cv2.rectangle(img,(int(dot[0]),int(dot[1])),(int(dot[0]+1),int(dot[1]+1)),(0,255,0))
        dots.append(dot)
    ptd2 = np.array(dots)
    ptd2 = np.reshape(ptd2,(-1,2))

    if len(img.shape) == 2:
        gray = copy.deepcopy(img)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for k in range(0,1):
        ndots = copy.deepcopy(ptd2)
        bbox = [np.min(ndots[:, 0]) - 2, np.min(ndots[:, 1]) - 2, np.max(ndots[:, 0]) + 2, np.max(ndots[:, 1]) + 2]

        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > img.shape[1] or bbox[3] > img.shape[0]:
            continue

        wbox = [np.min(ndots[18:, 0]), np.min(ndots[18:, 1]), np.max(ndots[18:, 0]), np.max(ndots[18:, 1])]
        nsize = max(wbox[2] - wbox[0], wbox[3] - wbox[1])
        side_range = [min([nsize * 1.6, img.shape[0], img.shape[1]]), min([nsize * 2.6, img.shape[0], img.shape[1]])]
        wmid = [(wbox[0] + wbox[2]) / 2.0, (wbox[1] + wbox[3]) / 2.0]
        midbox = [wmid[0] - nsize / 4.0, wmid[1] - nsize / 4.0, wmid[0] + nsize / 4.0, wmid[1] + nsize / 4.0]

        x, y, side = toCropRandom(gray, midbox, side_range, bbox)

        if x < 0 or y < 0:
            continue
        crop = gray[y:y + side, x:x + side]
        ndots[:, 0] -= x
        ndots[:, 1] -= y


        crop = cv2.resize(crop, (256, 256))
        scale = 256.0 / side
        ndots *= scale
 
        fname = save_root + str(nums) + '_' +  fidname  + '_'  + str(k)+ '_' + str(ans) + '.bmp'
        ans = ans + 1
        print(fname)
        cv2.imwrite(fname, crop)
        fw.write(fname+getTagStr(ndots)+'\n') 

fw.close()
