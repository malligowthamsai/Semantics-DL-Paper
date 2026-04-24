import sys
import torch.backends.cudnn as cudnn
sys.path.append('../')
sys.path.append('../data_process/')

import numpy as np
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import torch.nn as nn
import torch.optim as optims
import torch.utils.data as data
from tqdm import tqdm


from time import time
from only_for_vessel_seg.data_process.data_load import ImageFolder, get_drive_data

from only_for_vessel_seg.train_test.losses import loss_ce
from only_for_vessel_seg.train_test.eval_test import val_vessel
from torch.utils.tensorboard import SummaryWriter
from only_for_vessel_seg.train_test.help_functions import platform_info, check_size
from only_for_vessel_seg.train_test.evaluations import threshold_by_otsu

from DSFM_Net import DSFM_Net
from only_for_vessel_seg import Constants

learning_rates = Constants.learning_rates
gcn_model = False

# adjust learning rate (poly)
def adjust_lr(optimizer, base_lr, iter, max_iter, power=0.9):
    lr = base_lr * (1 - float(iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def update_lr1(optimizer, old_lr, ratio):
    for param_group in optimizer.param_groups:
        param_group['lr'] = old_lr / ratio
    print('update learning rate: %f -> %f' % (old_lr, old_lr / ratio))
    return old_lr / ratio


def update_lr2(epoch, optimizer, total_epoch=Constants.TOTAL_EPOCH):
    new_lr = learning_rates * (1 - epoch / total_epoch)
    for p in optimizer.param_groups:
        p['lr'] = new_lr


def optimizer_net(net, optimizers, criterion, images, masks,ch):
     optimizers.zero_grad()
     pred, pred1, pred2 = net(images)
     pred = net(images)
     loss0 = loss_ce(pred, masks, criterion, ch)
     loss1 = loss_ce(pred1, masks, criterion, ch)
     loss2 = loss_ce(pred2, masks, criterion, ch)
     loss = loss0 + loss1 + loss2
     loss.backward()
     optimizers.step()
     return pred, loss


def visual_preds(preds, is_preds=True):  # This for multi-classification
    rand_arr = torch.rand(size=(preds.size()[1], preds.size()[2], 3))
    color_preds = torch.zeros_like(rand_arr)
    outs = preds.permute((1, 2, 0))  # N H W C
    if is_preds is True:
        outs_one_hot = torch.argmax(outs, dim=2)
    else:
        outs_one_hot = outs.reshape((preds.size()[1], preds.size()[2]))
    for H in range(0, preds.size()[1]):
        for W in range(0, preds.size()[2]):
            if outs_one_hot[H, W] == 1:
                color_preds[H, W, 0] = 255
            if outs_one_hot[H, W] == 2:
                color_preds[H, W, 1] = 255
            if outs_one_hot[H, W] == 3:
                color_preds[H, W, 2] = 255
            if outs_one_hot[H, W] == 4:
                color_preds[H, W, 0] = 255
                color_preds[H, W, 1] = 255
                color_preds[H, W, 2] = 255
    return color_preds.permute((2, 0, 1))


def train_model(learning_rates):

    writer = SummaryWriter(comment=f"MyDRIVETrain01", flush_secs=1)
    tic = time()
    loss_lists = []
    no_optim = 0
    total_epoch = Constants.TOTAL_EPOCH
    train_epoch_best_loss = Constants.INITAL_EPOCH_LOSS
    ch = Constants.BINARY_CLASS
    criterion = nn.BCELoss()
    net = DSFM_Net(1, ch).to(device)  # 1
    optimizers = optims.Adam(net.parameters(), lr=learning_rates, betas=(0.9, 0.999))
    trains, val = get_drive_data()
    dataset = ImageFolder(trains[0], trains[1])
    data_loader = data.DataLoader(dataset, batch_size= Constants.BATCH_SIZE, shuffle=True, num_workers=0)
    rand_img, rand_label, rand_pred = None, None, None
    for epoch in range(1, total_epoch + 1):
        net.train(mode=True)
        data_loader_iter = iter((data_loader))
        train_epoch_loss = 0
        index = 0
        for img, mask in data_loader_iter:
            # check_size(img, mask, mask)
            img = img.to(device)
            mask = mask.to(device)
            pred, train_loss = optimizer_net(net, optimizers, criterion, img, mask, ch)
            train_epoch_loss += train_loss.item()
            index = index + 1
            if np.random.rand(1) > 0.4 and np.random.rand(1) < 0.8:
                rand_img, rand_label, rand_pred = img, mask, pred

        train_epoch_loss = train_epoch_loss / len(data_loader_iter)
        writer.add_scalar('Train/loss', train_epoch_loss, epoch)
        if ch ==1:      # for [N,1,H,W]
            rand_pred_cpu = rand_pred[0, :, :, :].detach().cpu().reshape((-1,)).numpy()
            # threshold = 0.5
            # rand_pred_cpu[rand_pred_cpu >= threshold] = 1
            # rand_pred_cpu[rand_pred_cpu <  threshold] = 0
            rand_pred_cpu = threshold_by_otsu(rand_pred_cpu)
            new_mask = rand_label[0, :, :, :].cpu().reshape((-1,)).numpy()
            writer.add_scalar('Train/acc', rand_pred_cpu[np.where(new_mask == rand_pred_cpu)].shape[0] / new_mask.shape[0], epoch)  # for [N,H,W,1]
        if ch ==2:      # for [N,2,H,W]
            new_mask = rand_label[0, :, :, :].cpu().reshape((-1,))
            new_pred = torch.argmax(rand_pred[0, :, :, :].permute((1, 2, 0)), dim=2).detach().cpu().reshape((-1,))
            t = new_pred[torch.where(new_mask == new_pred)].size()[0]
            writer.add_scalar('Train/acc', t / new_pred.size()[0], epoch)

        platform_info(epoch, tic, train_epoch_loss, Constants.IMG_SIZE, optimizers)
        if epoch % 10 == 1:
            writer.add_image('Train/image_origins', rand_img[0, :, :, :], epoch)
            writer.add_image('Train/image_labels', rand_label[0, :, :, :], epoch)
            if ch == 1:  # for [N,1,H,W]
                writer.add_image('Train/image_predictions', rand_pred[0, :, :, :], epoch)
            if ch == 2:  # for [N,2,H,W]
                  writer.add_image('Train/image_predictions', torch.unsqueeze(torch.argmax(rand_pred[0, :, :, :], dim=0), 0),
                             epoch)
        update_lr2(epoch, optimizers)  # modify  lr

        print('************ start to validate current model {}.iter performance ! ************'.format(epoch))
        acc, sen, f1score, val_loss = val_vessel(net, val[0], val[1], val[0].shape[0], epoch)
        writer.add_scalar('Val/accuracy', acc, epoch)
        writer.add_scalar('Val/sensitivity', sen, epoch)
        writer.add_scalar('Val/f1score', f1score, epoch)
        writer.add_scalar('Val/val_loss', val_loss, epoch)

        model_name = Constants.saved_path + "{}.iter3".format(epoch)
        torch.save(net, model_name)

    print('***************** Finish training process ***************** ')

if __name__ == '__main__':
    train_model(learning_rates)
    pass

