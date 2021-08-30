import numpy as np
import cv2,torch
from scipy import ndimage
from sklearn.metrics.pairwise import pairwise_distances

def dice_loss(masks, labels, is_average=True):
    """
    dice loss
    :param masks:
    :param labels:
    :return:
    """
    num = labels.size(0)

    m1 = masks.view(num, -1)
    m2 = labels.view(num, -1)

    intersection = (m1 * m2)

    score = (2 * intersection.sum(1)) / (m1.sum(1) + m2.sum(1)+1.0)
    if is_average:
        return score.sum() / num
    else:
        return score
def dice_ratio(masks, labels, is_average=True):
    """
    dice ratio
    :param masks:
    :param labels:
    :return:
    """
    masks = masks.cpu()
    labels = labels.cpu()
    
    m1 = masks.flatten()
    m2 = labels.flatten().float()

    intersection = m1 * m2
    score = (2 * intersection.sum()) / (m1.sum() + m2.sum()+1e-6)

    pre = intersection.sum() / np.max([m2.sum(), 1])
    rec = intersection.sum() / np.max([m1.sum(), 1])

    return score#, pre, rec    
def dice_mc(masks, labels, classes):
    
    
    num = labels.size(0)

    class_dice = torch.zeros(num)
    per_class_dice = torch.zeros(num,classes)
    per_class_cnt = torch.zeros(num,classes)

    total_insect = 0.0
    total_pred = 0.0
    total_labs = 0.0
    
    for i in range(num):
        for n in range(1,classes):
            if (labels[i]==n).sum():
                pred = (masks[i]==n)
                labs = (labels[i]==n)
                insect = pred*labs
                per_class_dice[i,n-1] =(2 * insect.sum()).float() / (pred.sum() + labs.sum()).float()
                per_class_cnt[i,n-1] +=1
             
                total_insect += insect.sum()
                total_pred += pred.sum()
                total_labs += labs.sum()
                
        class_dice[i] = (2*total_insect).float()/ (total_pred + total_labs).float()
        
    aver_dice = class_dice.sum()/num
    per_class_dice = per_class_dice.sum(0)/(per_class_cnt.sum(0)+1e-5)
    return aver_dice,per_class_dice

def dice_m(masks, labels, classes):
    
    
    num = labels.size(0)

    m1 = masks.view(num, -1)
    m2 = labels.view(num, -1)


    class_dice = torch.zeros(num)
    per_class_dice = torch.zeros(num,classes)
    m1_cnt = torch.zeros(num,classes)
    m2_cnt = torch.zeros(num,classes)
    insect_cnt = torch.zeros(num,classes)
    
    for i in range(num):
        for j in range(m1.shape[1]):
            if m1[i,j]!=0:
                if m1[i,j]==m2[i,j]:
                    insect_cnt[i,m1[i,j]-1] += 1 
                m1_cnt[i,m1[i,j]-1] += 1
            if m2[i,j]!=0:
                m2_cnt[i,m2[i,j]-1] += 1
        
        
        per_class_dice[i] =(2 * insect_cnt[i]) / (m1_cnt[i] + m2_cnt[i])
             
        class_dice[i] = (2*insect_cnt[i].sum())/ (m1_cnt[i].sum() + m2_cnt[i].sum())
    class_dice = class_dice.sum()/num
    per_class_dice = per_class_dice.sum(0)/num
    return class_dice,per_class_dice



def hausdorff_mad_distance(set1, set2, max_ahd=np.inf):
    """
    Compute the Averaged Hausdorff Distance function
     between two unordered sets of points (the function is symmetric).
     Batches are not supported, so squeeze your inputs first!
    :param set1: Array/list where each row/element is an N-dimensional point.
    :param set2: Array/list where each row/element is an N-dimensional point.
    :param max_ahd: Maximum AHD possible to return if any set is empty. Default: inf.
    :return: The Hausdorff Distance and Mean Absolute Distance between set1 and set2.
    """

    if len(set1) == 0 or len(set2) == 0:
        return max_ahd

    set1 = np.array(set1.cpu())
    set2 = np.array(set2.cpu())

    assert set1.ndim == 2, 'got %s' % set1.ndim
    assert set2.ndim == 2, 'got %s' % set2.ndim

    assert set1.shape[1] == set2.shape[1], \
        'The points in both sets must have the same number of dimensions, got %s and %s.'\
        % (set2.shape[1], set2.shape[1])

    d2_matrix = pairwise_distances(set1, set2, metric='euclidean')

    d12 = np.min(d2_matrix, axis=0)
    d21 = np.min(d2_matrix, axis=1)
    #print(d12.size,d21.size)

    hd = np.max([np.max(d12),np.max(d21),0])

    # sorted_d12 = np.sort(d12)
    # sorted_d21 = np.sort(d21)
    # num12 = np.int(np.round(np.size(d12) * 0.95))
    # num21 = np.int(np.round(np.size(d21) * 0.95))
    #
    # mhd = np.max([sorted_d12[num12], sorted_d21[num21]])

    # mad = 0.5*(np.average(d12)+np.average(d21))

    return hd#, mhd#, mad

def acc(masks, labels):

    # labels = labels.cpu().numpy()
    # masks = masks.cpu().numpy()
    
    m1 = masks.flatten()
    m2 = labels.flatten()
    
    same = (m1 == m2).sum().float()
    diff = (m1 != m2).sum().float()
       
    intersection = m1 * m2
    same1 = intersection.sum()
    same0 = same - intersection.sum()
    acc = same/m2.size(0)
    return acc,same,m2.size(0)#,same0,same1,diff

def acc_test(masks, labels, masks_con):

    masks1 = masks.flatten()
    lab1 = labels.flatten()

    masks1 = masks1.cpu().numpy()
    loc = np.argwhere(masks1==0)
    masks2 = masks_con.flatten()[loc]
    # masks3 = masks_rad.flatten()[loc]

    # print(masks2.max(),masks2.min(),masks2.sum())
    #     # print(masks3.max(),masks3.min(), masks3.sum())
    lab2 = lab1[loc]
    # print(type(masks2), type(lab2))
    m1 = masks2
    m2 = lab2

    same = (m1 == m2).sum().float()
    intersection = m1 * m2
    same1 = intersection.sum()#/len(m2)#same
    same0 = (same - intersection.sum())#/len(m2)#same

    acc = same#/len(m2)
    dice = 2*intersection.sum().float()/((m1.sum() + m2.sum()+1.0))

    mis0 = ((m1 != m2) & (m2 == 1)).sum().float()#/len(m2)
    mis1 = ((m1 != m2) & (m2 == 0)).sum().float()#/len(m2)

    # #2
    # same = 0
    # same0 = 0
    # same1 = 0
    # pred1 = 0
    # pred0 = 0
    # lab1 = 0
    # lab0 = 0
    # for i in len(loc):
    #     if masks_con[i]==lab[i]:
    #         same +=1
    #         if  masks_con[i] == 1:
    #             same1 +=1
    #         else:
    #             same0 +=1
    #     elif masks_con[i]==1:
    #         pred1 +=1
    #     else:
    #         lab1 +=1

    # acc = same/len(loc).sum()

    # dice = 2*same1/(pred1+lab1)
    return acc,dice,same0,same1,mis0,mis1,len(m1)#,diff


def acc_m(masks, labels, masks_con):
    masks1 = masks.flatten()
    lab1 = labels.flatten()

    masks1 = masks1.cpu().numpy()
    loc = np.argwhere(masks1 == 0)
    masks2 = masks_con.flatten()[loc]
    # masks3 = masks_rad.flatten()[loc]

    # print(masks2.max(),masks2.min(),masks2.sum())
    #     # print(masks3.max(),masks3.min(), masks3.sum())
    lab2 = lab1.flatten()[loc]

    m1 = masks2
    m2 = lab2

    same = (m1 == m2).sum().float()
    intersection = m1 * m2
    same1 = intersection.sum()/same
    same0 = (same - intersection.sum())/same

    acc = same#/len(m2)
    dice = 2 * intersection.sum().float() / ((m1.sum() + m2.sum() + 1.0))

    mis0 = ((m1 != m2) & (m2 == 1)).sum().float()  # /len(m2)
    mis1 = ((m1 != m2) & (m2 == 0)).sum().float()  # /len(m2)

    # #2
    # same = 0
    # same0 = 0
    # same1 = 0
    # pred1 = 0
    # pred0 = 0
    # lab1 = 0
    # lab0 = 0
    # for i in len(loc):
    #     if masks_con[i]==lab[i]:
    #         same +=1
    #         if  masks_con[i] == 1:
    #             same1 +=1
    #         else:
    #             same0 +=1
    #     elif masks_con[i]==1:
    #         pred1 +=1
    #     else:
    #         lab1 +=1

    # acc = same/len(loc).sum()

    # dice = 2*same1/(pred1+lab1)
    return acc, dice, same0, same1#, mis0, mis1, len(m1)  # ,diff


def pre_rec(masks, labels):
    """
    dice ratio
    :param masks:
    :param labels:
    :return:
    """
    m1 = masks.flatten()
    m2 = labels.flatten().float()

    intersection = m1 * m2

    pre = intersection.sum() / (m1.sum()+1e-6)
    rec = intersection.sum() / (m2.sum()+1e-6)

    return pre, rec