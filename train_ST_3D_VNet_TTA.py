import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils1.loss
from dataset.make_dataset import make_data_3d
from dataset.pancreas import Pancreas
from test_util import test_calculate_metric
from utils1 import statistic, ramps
from utils1.loss import DiceLoss
from vnet import VNet

import logging
import sys

"""
Test Time Augmentation
Average multi-runs of outputs of test time augmented images

Reference:
Combalia, Marc, et al. "Uncertainty estimation in deep neural networks for dermoscopic image classification." CVPRW 2020.
"""

# 2, 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,4,5'
# Parameters
num_class = 2
base_dim = 8

batch_size = 8
lr = 0.01
beta1, beta2 = 0.5, 0.999
w_con = torch.FloatTensor([1, 5])
w_rad = torch.FloatTensor([5, 1])

# log settings & test
pretraining_epochs = 60
self_training_epochs = 101
thres = 0.5

pretrain_save_step = 20
st_save_step = 20
pred_step = 10

res_dir = 'result/pancreas_final_VNet/'
split_name = 'pancreas'
data_root = '/data/DataSets/pancreas_pad25'
cost_num = 3

alpha = 0.99
consistency = 0.1
consistency_rampup = 1

logging.basicConfig(filename=res_dir + "log.txt", level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info('\n\n New Exp :')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        return self

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count
        return self


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return ramps.sigmoid_rampup(epoch, consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)


def create_model(ema=False):
    net = nn.DataParallel(VNet(has_dropout=False))
    # net = nn.DataParallel(VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True))
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def get_model_and_dataloader():
    """Net & optimizer"""
    net = create_model()
    ema_net = create_model(ema=True).cuda()
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta1, beta2))
    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)

    """Loading Dataset"""
    logging.info("loading dataset")
    trainset_lab = Pancreas(data_root, split_name, split='train_lab')
    lab_loader = DataLoader(trainset_lab, batch_size=batch_size, shuffle=False, num_workers=0)

    trainset_unlab = Pancreas(data_root, split_name, split='train_unlab', no_crop=True)
    unlab_loader = DataLoader(trainset_unlab, batch_size=1, shuffle=False, num_workers=0)

    testset = Pancreas(data_root, split_name, split='test')
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    logging.info(len(lab_loader.dataset), len(unlab_loader.dataset), len(test_loader.dataset))
    return net, ema_net, optimizer, lab_loader, unlab_loader, test_loader



def save_net_opt(net, optimizer, path, epoch):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, str(path))


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])
    logging.info('Loaded from {}'.format(path))


def pretrain(net, ema_net, optimizer, lab_loader, unlab_loader, test_loader, start_epoch=1):
    save_path = Path(res_dir) / 'pretrain_con_{}_consistency_{}'.format(w_con[1].item(), consistency)
    save_path.mkdir(exist_ok=True)
    logging.info("Save path : ", save_path)

    writer = SummaryWriter(str(save_path), filename_suffix=time.strftime('_%Y-%m-%d_%H-%M-%S'))

    DICE = DiceLoss(nclass=2)
    CE_con = nn.CrossEntropyLoss(weight=w_con.cuda())
    CE_rad = nn.CrossEntropyLoss(weight=w_rad.cuda())
    maxdice1 = 0

    iter_num = 0
    for epoch in tqdm(range(start_epoch, pretraining_epochs + 1), ncols=70):
        """Testing"""
        if epoch % pretrain_save_step == 0:
            # maxdice, _ = test(net, unlab_loader, maxdice, max_flag)
            val_dice, maxdice1, max_flag = test(net, test_loader, maxdice1)

            writer.add_scalar('pretrain/test_dice', val_dice, epoch)

            save_net_opt(net, optimizer, save_path / ('%d.pth' % epoch), epoch)
            logging.info('Save model : {}'.format(epoch))
            if max_flag:
                save_net_opt(net, optimizer, save_path / 'best.pth', epoch)
                save_net_opt(ema_net, optimizer, save_path / 'best_ema.pth', epoch)

        train_loss, train_dice, ce_loss_, dice_loss_, loss_con_, loss_rad_ = \
            AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        net.train()
        for step, (img, lab) in enumerate(lab_loader):
            img, lab = img.cuda(), lab.cuda()
            out = net(img)

            ce_loss = F.cross_entropy(out[0], lab)
            dice_loss = DICE(out[0], lab)
            loss = (ce_loss + dice_loss) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            masks = get_mask(out[0])
            train_dice.update(statistic.dice_ratio(masks, lab), 1)
            train_loss.update(loss.item(), 1)
            ce_loss_.update(ce_loss.item(), 1)
            dice_loss_.update(dice_loss.item(), 1)

            logging.info('epoch : %d, step : %d, train_loss: %.4f, train_dice: %.4f' % (epoch, step, train_loss.avg, train_dice.avg))

            writer.add_scalar('pretrain/loss_ce', ce_loss_.avg, epoch * len(lab_loader) + step)
            writer.add_scalar('pretrain/loss_dice', dice_loss_.avg, epoch * len(lab_loader) + step)
            writer.add_scalar('pretrain/loss_all', train_loss.avg, epoch * len(lab_loader) + step)
            writer.add_scalar('pretrain/train_dice', train_dice.avg, epoch * len(lab_loader) + step)
            update_ema_variables(net, ema_net, alpha, step)
        writer.flush()


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


def get_mask(out, thres=0.5):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).float()
    masks = masks[:, 1, :, :].contiguous()
    return masks


def st_train(net, ema_net, optimizer, lab_loader, unlab_loader, test_loader):
    save_path = Path(res_dir) / 'con_{}_consistency_{}_st{}_TTA'.format(w_con[1].item(), consistency, st_save_step)
    save_path.mkdir(exist_ok=True)
    logging.info("Save path : ", save_path)

    writer = SummaryWriter(str(save_path), filename_suffix=time.strftime('_%Y-%m-%d_%H-%M-%S'))
    pretrained_path = Path(res_dir) / 'pretrain_con_{}_consistency_{}'.format(w_con[1].item(), 1)

    load_net_opt(net, optimizer, pretrained_path / 'best.pth')
    load_net_opt(ema_net, optimizer, pretrained_path / 'best.pth')

    # load_net_opt(net, optimizer, save_path / 'best.pth')
    # load_net_opt(ema_net, optimizer, save_path / 'best.pth')

    consistency_criterion = utils1.loss.softmax_mse_loss
    CE_CA = nn.CrossEntropyLoss(reduction='none')
    DICE = DiceLoss(nclass=2)
    CE = nn.CrossEntropyLoss()

    maxdice = 0
    maxdice1 = 0

    iter_num = 0
    new_loader, plab_dice = pred_unlabel(net, unlab_loader)
    writer.add_scalar('acc/plab_dice', plab_dice, 0)

    for epoch in tqdm(range(1, self_training_epochs)):
        logging.info('')
        writer.flush()

        if epoch % st_save_step == 0:
            """Testing"""
            # val_dice, maxdice, _ = test(net, unlab_loader, maxdice)
            val_dice, maxdice1, max_flag = test(net, test_loader, maxdice1)
            writer.add_scalar('acc/val_dice', val_dice, epoch)

            """Save model"""
            if epoch > 100:
                save_net_opt(net, optimizer, str(save_path / ('{}.pth'.format(epoch))), epoch)
                logging.info('Save model : {}'.format(epoch))

            if max_flag:
                save_net_opt(net, optimizer, str(save_path / 'best.pth'), epoch)

        if epoch % pred_step == 0:
            new_loader, plab_dice = pred_unlabel(net, unlab_loader)
            writer.add_scalar('acc/plab_dice', plab_dice, epoch)

        train_loss, train_loss1, train_loss2, train_loss3, train_dice, unlab_dice = \
            AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        loss_ce1_, loss_con1_, loss_rad1_, dice_loss1_, loss_ce2_, loss_con2_, loss_rad2_, dice_loss2_ = \
            AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        lab_rad_dice, unlab_rad_dice, lab_con_dice, unlab_con_dice = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        net.train()
        ema_net.train()
        for step, (data1, data2) in enumerate(zip(lab_loader, new_loader)):
            img1, lab1 = data1
            img1, lab1 = img1.cuda(), lab1.long().cuda()
            img2, plab2, mask, lab2 = data2
            img2, plab2, mask = img2.cuda(), plab2.long().cuda(), mask.float().cuda()
            # plab2 = lab2.cuda()

            '''Supervised Loss'''
            out1 = net(img1)
            dice_loss1 = DICE(out1[0], lab1)
            loss_ce1 = CE(out1[0], lab1)
            supervised_loss = (loss_ce1 + dice_loss1) / 2

            # TODO : For Ablation Study
            # mask = torch.zeros_like(mask).cuda(mask.device).float()

            '''Certain Areas'''
            out2 = net(img2)
            dice_loss2 = DICE(out2[0], plab2, mask)
            loss_ce2 = (CE_CA(out2[0], plab2) * mask).sum() / (mask.sum() + 1e-16)  #
            certain_loss = (loss_ce2 + dice_loss2) / 2

            '''Uncertain Areas---Mean Teacher'''
            mask = (1 - mask).unsqueeze(1)
            with torch.no_grad():
                out_ema = ema_net(img2)
            consistency_weight = consistency * get_current_consistency_weight(epoch)
            consistency_dist1 = consistency_criterion(out2[0], out_ema[0])
            const_loss1 = consistency_weight * ((consistency_dist1 * mask).sum() / (mask.sum() + 1e-16))
            uncertain_loss = const_loss1
            # logging.info(uncertain_loss)
            loss = supervised_loss + certain_loss + uncertain_loss  # uncertain_loss * 0.3 #+ certain_loss*0.5

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                update_ema_variables(net, ema_net, alpha, iter_num + len(lab_loader) * pretraining_epochs)
                iter_num = iter_num + 1

                mask1 = get_mask(out1[0])
                mask2 = get_mask(out2[0])

                train_dice.update(statistic.dice_ratio(mask1, lab1), 1)
                unlab_dice.update(statistic.dice_ratio(mask2, lab2), 1)

                lab_rad_dice.update(statistic.dice_ratio(get_mask(out1[2]), lab1), 1)
                unlab_rad_dice.update(statistic.dice_ratio(get_mask(out2[2]), lab2), 1)
                lab_con_dice.update(statistic.dice_ratio(get_mask(out1[1]), lab1), 1)
                unlab_con_dice.update(statistic.dice_ratio(get_mask(out2[1]), lab2), 1)

                train_loss.update(loss.item(), 1)
                train_loss3.update(uncertain_loss.item(), 1)

                loss_ce1_.update(loss_ce1.item(), 1)
                dice_loss1_.update(dice_loss1.item(), 1)
                train_loss1.update(supervised_loss.item(), 1)

                loss_ce2_.update(loss_ce2.item(), 1)
                dice_loss2_.update(dice_loss2.item(), 1)
                train_loss2.update(certain_loss.item(), 1)

            logging.info('epoch : {}, '
                         'lab_loss: {:.4f}, unlab_certain_loss: {:.4f}, unlab_uncertain_loss: {:.4f}, '
                         'train_loss: {:.4f}, train_dice: {:.4f}, unlab_dice: {:.4f}, '
                         'lab_rad: {:.4f}, lab_con: {:.4f}, unlab_rad: {:.4f}, unlab_con: {:.4f}'.format(
                epoch,
                train_loss1.val, train_loss2.val, train_loss3.val,
                train_loss.val, train_dice.val, unlab_dice.val,
                lab_rad_dice.val, lab_con_dice.val, unlab_rad_dice.val, unlab_con_dice.val))

        writer.add_scalar('supervised_loss/all', train_loss1.avg, epoch)
        writer.add_scalar('supervised_loss/ce', loss_ce1_.avg, epoch)
        writer.add_scalar('supervised_loss/rad', loss_rad1_.avg, epoch)
        writer.add_scalar('supervised_loss/con', loss_con1_.avg, epoch)
        writer.add_scalar('supervised_loss/dice', dice_loss1_.avg, epoch)

        writer.add_scalar('unsup_loss/certain_all', train_loss2.avg, epoch)
        writer.add_scalar('unsup_loss/certain_ce', loss_ce2_.avg, epoch)
        writer.add_scalar('unsup_loss/certain_rad', loss_rad2_.avg, epoch)
        writer.add_scalar('unsup_loss/certain_con', loss_con2_.avg, epoch)
        writer.add_scalar('unsup_loss/certain_dice', dice_loss2_.avg, epoch)

        writer.add_scalar('unsup_loss/uncertain_loss', train_loss3.avg, epoch)

        writer.add_scalar('acc/lab_dice', train_dice.avg, epoch)
        writer.add_scalar('acc/unlab_dice', unlab_dice.avg, epoch)
        writer.add_scalar('acc/unlab_rad_dice', unlab_rad_dice.avg, epoch)
        writer.add_scalar('acc/unlab_con_dice', unlab_con_dice.avg, epoch)
        writer.add_scalar('acc/lab_con_dice', lab_con_dice.avg, epoch)
        writer.add_scalar('acc/lab_rad_dice', lab_rad_dice.avg, epoch)


class RandCrop():
    def __init__(self, size):
        self.size = size

    def get_pos(self, shape):
        w, h, d = shape
        if w == self.size[0]:
            i = 0
        else:
            i = random.randint(0, w - self.size[0])

        if h == self.size[1]:
            j = 0
        else:
            j = random.randint(0, h - self.size[1])

        if w == self.size[2]:
            k = 0
        else:
            k = random.randint(0, d - self.size[2])
        return i, j, k

    def crop(self, img):
        pos = self.get_pos(img.shape[2:])
        cropped_img = img[:, :, pos[0]:pos[0] + self.size[0], pos[1]:pos[1] + self.size[1], pos[2]:pos[2] + self.size[2]]
        return cropped_img, pos

    def put_back(self, img, target, pos):
        target[:, :, pos[0]:pos[0] + self.size[0], pos[1]:pos[1] + self.size[1], pos[2]:pos[2] + self.size[2]] += img
        return target


random_crop = RandCrop((96, 96, 96))


@torch.no_grad()
def pred_unlabel(net, pred_loader):
    logging.info('Starting predict unlab')
    unimg, unlab, unmask, labs = [], [], [], []
    plab_dice = 0

    for (step, data) in enumerate(pred_loader):
        img, lab = data
        img, lab = img.cuda(), lab.cuda()

        logits = net(img)[0]
        out, num = logits, torch.ones_like(logits)
        for i in range(20):
            aug_img, pos = random_crop.crop(img)
            logits = net(aug_img)[0]
            random_crop.put_back(logits, out, pos)
            random_crop.put_back(torch.ones_like(logits).cuda(), num, pos)
        out = out / num

        plab = get_mask(out, thres=0.5)
        prob = F.softmax(out, 1)
        mask = ((prob >= 0.8).sum(1) > 0).float()

        unimg.append(img)
        unlab.append(plab)
        unmask.append(mask)
        labs.append(lab)

        plab_dice += statistic.dice_ratio(plab, lab)
    plab_dice /= len(pred_loader)
    logging.info('Pseudo label dice : {}'.format(plab_dice))
    new_loader1 = DataLoader(make_data_3d(unimg, unlab, unmask, labs), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    # new_loader2 = DataLoader(make_data(unimg2, unlab2), batch_size=batch_size, shuffle=True, num_workers=0)
    return new_loader1, plab_dice


@torch.no_grad()
def test(net, val_loader, maxdice=0):
    metrics = test_calculate_metric(net, val_loader.dataset)
    val_dice = metrics[0]

    if val_dice > maxdice:
        maxdice = val_dice
        max_flag = True
    else:
        max_flag = False
    logging.info('Evaluation : val_dice: %.4f, val_maxdice: %.4f' % (val_dice, maxdice))
    return val_dice, maxdice, max_flag


if __name__ == '__main__':
    # set_random_seed(1337)
    net, ema_net, optimizer, lab_loader, unlab_loader, test_loader = get_model_and_dataloader()
    # load model
    # net.load_state_dict(torch.load(res_dir + '/model/best.pth'))
    pretrained_path = Path(res_dir) / 'pretrain_con_{}_consistency_{}'.format(w_con[1].item(), consistency)

    # load_net_opt(net, optimizer, pretrained_path / 'best.pth')
    # load_net_opt(ema_net, optimizer, pretrained_path / 'best.pth')
    # pretrain(net, ema_net, optimizer, lab_loader, unlab_loader, test_loader, start_epoch=1)

    st_train(net, ema_net, optimizer, lab_loader, unlab_loader, test_loader)

    logging.info(count_param(net))
