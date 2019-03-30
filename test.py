import hg_net as net
import torch
import test_data_loader as loader
import numpy as np
import cv2
import shutil
import os
import torch.nn as nn
from torch.autograd import Variable
#import fy_net as net
model = net.FAN(2)
#model_path = './model/model_29_6762.pkl'
model_path = './good_model/model_80_13525.pkl'
#model_path = './model/model_11_3387.pkl'
model.load_state_dict(torch.load(model_path))
save_path = './small_out_imgs/'
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)
device = torch.cuda.set_device(2)
model.cuda(device)
sl1_criterion = nn.SmoothL1Loss().cuda()
test_list = './256_Data_test/data_tag.list'
test_data = loader.getDataFromList(test_list)
test_dataset = loader.DataLoader(test_data, 256, 68)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False, num_workers = 8, pin_memory = True)
def test():
    ok = True
    Min = 100
    Max = 0
    Loss = 0
    model.eval()
    ans = 0
    for batch_idx, data in enumerate(test_loader):
        img, tag, dots, img_name = data
        img, tag, dots = Variable(img).cuda(async = True), Variable(tag).cuda(async = True), Variable(dots).cuda(async = True)
        out, reg_outs = model(img)
        colors = [(255, 255, 0), (0, 255, 255)]
        for k, reg in enumerate(reg_outs):
            # dots = Variable(dots).cuda(async = True)
            loss = sl1_criterion(reg, dots)
            if k == 0:
                all_loss = loss
            else:
                all_loss += loss
            Loss += loss.data.cpu().numpy()
            reg = reg.cpu().data
            reg = np.array(reg)
            reg *= 4.0
            reg += 32.0
            reg *= 4.0
            reg = np.reshape(reg, (-1, 2))
            test_img = cv2.imread(img_name[0])

            if len(test_img.shape) == 2:
                test_img = copy.deepcopy(test_img)
            else:
                test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

            for i in range(0, reg.shape[0]):
                cv2.rectangle(test_img, (int(reg[i, 0]) - 1, int(reg[i, 1]) - 1), (int(reg[i, 0] + 1), int(reg[i, 1]) + 1), colors[k])
            out_imgs = save_path + str(ans) + '.bmp'
            cut = 0
            ans += 1
            cv2.imwrite(out_imgs, test_img)

        All_loss = all_loss.data.cpu().numpy()
        print('loss: {:.6f}'.format(float(all_loss)))
        Min = min(Min, All_loss)
        Max = max(Max, All_loss)
    ans *= 1.0
    print(Loss / ans)
    print('[{:.6f}, {:.6f}]'.format(float(Min), float(Max)))

test()
