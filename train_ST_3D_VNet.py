import os

import torch
from torch.backends import cudnn
from tqdm import tqdm
from preprocess.io_ import mkdir
from utils1.loss import DiceLoss, softmax_mse_loss
from utils1.utils import *
from test_util import test_calculate_metric
from utils1.visualize import show_graphs

"""Global Variables"""
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

data_root, split_name = '/data/DataSets/pancreas_pad25', 'pancreas'

result_dir = 'result/pancreas_VNet_{}/'
mkdir(result_dir)

w_con, w_rad = torch.FloatTensor([1, 5]).cuda(), torch.FloatTensor([5, 1]).cuda()
batch_size, lr = 8, 1e-3
pretraining_epochs, self_training_epochs = 60, 200
pretrain_save_step, st_save_step, pred_step = 20, 20, 5
alpha, consistency, consistency_rampup = 0.99, 0.1, 40

consistency_criterion = softmax_mse_loss
CE = nn.CrossEntropyLoss()
CE_con = nn.CrossEntropyLoss(weight=w_con)
CE_rad = nn.CrossEntropyLoss(weight=w_rad)
CE_r = nn.CrossEntropyLoss(reduction='none')
CE_con_r = nn.CrossEntropyLoss(reduction='none', weight=w_con)
CE_rad_r = nn.CrossEntropyLoss(reduction='none', weight=w_rad)
DICE = DiceLoss(nclass=2)

logger = None


def pretrain(net, optimizer, lab_loader, test_loader):
    """
    Pretrain Stage
    """

    """Create Path"""
    save_path = Path(result_dir) / 'pretrain_con_{}'.format(w_con[1].item(), consistency)
    save_path.mkdir(exist_ok=True)

    """Create logger and measures"""
    global logger
    logger, writer = config_log(save_path, tensorboard=True)
    logger.info("Pretrain, save path : " + str(save_path))

    max_dice = 0
    measures = PretrainMeasures(writer, logger)
    for epoch in tqdm(range(1, pretraining_epochs + 1), ncols=70):
        measures.reset()
        """Testing"""
        if epoch % pretrain_save_step == 0:
            avg_metric = test_calculate_metric(net, test_loader.dataset)
            logger.info('average metric is {}'.format(avg_metric))
            val_dice = avg_metric[0]

            save_net_opt(net, optimizer, save_path / 'best.pth', epoch)
            if val_dice > max_dice:
                max_dice = val_dice

            writer.add_scalar('pretrain/test_dice', val_dice, epoch)
            logger.info('Evaluation : val_dice: %.4f, val_maxdice: %.4f' % (val_dice, max_dice))
            save_net_opt(net, optimizer, save_path / ('%d.pth' % epoch), epoch)

        """Training"""
        net.train()
        for step, (img, lab) in enumerate(lab_loader):
            img, lab = img.cuda(), lab.cuda()
            out = net(img)

            ce_loss = F.cross_entropy(out[0], lab)
            dice_loss = DICE(out[0], lab)
            loss_con = CE_con(out[1], lab)
            loss_rad = CE_rad(out[2], lab)
            loss = (ce_loss + dice_loss + loss_con + loss_rad) / 4

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            measures.update(out[0], lab, ce_loss, dice_loss, loss_con, loss_rad, loss)
            measures.log(epoch, epoch * len(lab_loader) + step)
        writer.flush()


def self_train(net, ema_net, optimizer, lab_loader, unlab_loader, test_loader):
    """
    Self Training Stage
    """

    """Create Path"""
    save_path = Path(result_dir) / 'w_{}_all*0.5_pred5'.format(w_con[1].item(), consistency, st_save_step)
    save_path.mkdir(exist_ok=True)

    """Create logger and measures"""
    global logger
    logger, writer = config_log(save_path, tensorboard=True)
    logger.info("Self-training, save path : {}".format(str(save_path)))
    measures = STMeasures(writer, logger)

    """Load Model"""
    pretrained_path = Path(result_dir) / 'pretrain_con_{}'.format(w_con[1].item(), consistency)
    load_net_opt(net, optimizer, pretrained_path / 'best.pth')
    load_net_opt(ema_net, optimizer, pretrained_path / 'best.pth')
    logger.info('Loaded from {}'.format(pretrained_path))

    max_dice = 0
    for epoch in tqdm(range(0, self_training_epochs + 1)):
        measures.reset()
        logger.info('')

        """Testing"""
        if epoch % st_save_step == 0 and epoch > 0:
            avg_metric = test_calculate_metric(net, test_loader.dataset)
            logger.info('average metric is {}'.format(avg_metric))
            val_dice = avg_metric[0]
            writer.add_scalar('val_dice', val_dice, epoch)

            save_net_opt(net, optimizer, str(save_path / 'best.pth'), epoch)
            """Save model"""
            if val_dice > max_dice:
                max_dice = val_dice

            logger.info('Evaluation : val_dice: %.4f, val_maxdice: %.4f' % (val_dice, max_dice))

        """Predict pseudo labels"""
        if epoch % pred_step == 0:
            logger.info('Starting predict unlab')
            new_loader, plab_dice = pred_unlabel(net, unlab_loader, batch_size)
            logger.info('Pseudo label dice : {}'.format(plab_dice))
            writer.add_scalar('dice/plab_dice', plab_dice, epoch)

        """Training"""
        net.train()
        ema_net.train()
        for step, ((img_l, lab_l), (img_unl, plab, mask, lab_unl)) in enumerate(zip(lab_loader, new_loader)):
            img_l, lab_l, img_unl, plab, mask = to_cuda([img_l, lab_l, img_unl, plab, mask])

            '''Supervised Loss'''
            out1 = net(img_l)
            dice_loss1 = DICE(out1[0], lab_l)
            loss_ce1 = CE(out1[0], lab_l).mean()
            loss_con1 = CE_con(out1[1], lab_l).mean()
            loss_rad1 = CE_rad(out1[2], lab_l).mean()
            supervised_loss = (loss_ce1 + dice_loss1 + loss_con1 + loss_rad1) / 4

            '''Certain Areas---Self Training'''
            out2 = net(img_unl)
            dice_loss2 = DICE(out2[0], plab, mask)
            loss_ce2 = (CE_r(out2[0], plab) * mask).sum() / (mask.sum() + 1e-16)  #
            loss_con2 = (CE_rad_r(out2[1], plab) * mask).sum() / (mask.sum() + 1e-16)  #
            loss_rad2 = (CE_con_r(out2[2], plab) * mask).sum() / (mask.sum() + 1e-16)  #
            certain_loss = (loss_ce2 + dice_loss2 + loss_con2 + loss_rad2) / 4

            '''Uncertain Areas---Mean Teacher'''
            mask = (1 - mask).unsqueeze(1)
            with torch.no_grad():
                out_ema = ema_net(img_unl)
            consistency_weight = consistency * get_current_consistency_weight(epoch, consistency_rampup)
            consistency_dist1 = consistency_criterion(out2[0], out_ema[0])
            const_loss1 = consistency_weight * ((consistency_dist1 * mask).sum() / (mask.sum() + 1e-16))
            consistency_dist2 = consistency_criterion(out2[1], out_ema[1])
            const_loss2 = consistency_weight * ((consistency_dist2 * mask).sum() / (mask.sum() + 1e-16))
            consistency_dist3 = consistency_criterion(out2[2], out_ema[2])
            const_loss3 = consistency_weight * ((consistency_dist3 * mask).sum() / (mask.sum() + 1e-16))
            uncertain_loss = (const_loss1 + const_loss2 + const_loss3) / 3

            loss = supervised_loss + (certain_loss + uncertain_loss) * 0.5

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_ema_variables(net, ema_net, alpha)

            measures.update(out1, out2, lab_l, lab_unl, loss,
                            supervised_loss, loss_ce1, loss_rad1, loss_con1, dice_loss1,
                            certain_loss, loss_ce2, loss_rad2, loss_con2, uncertain_loss, )
            measures.log(epoch)
        measures.write_tensorboard(epoch)


if __name__ == '__main__':
    try:
        net, ema_net, optimizer, lab_loader, unlab_loader, test_loader = get_model_and_dataloader(data_root, split_name, batch_size, lr, res18=False)
        pretrain(net, optimizer, lab_loader, test_loader)
        self_train(net, ema_net, optimizer, lab_loader, unlab_loader, test_loader)
    except Exception as e:
        logger.exception("Exception found!!!")
