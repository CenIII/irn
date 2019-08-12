
import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
import imageio
from misc import torchutils, imutils
import tqdm
import copy
import pickle

def one_loop(dataset, labels, args, thres):
    preds = []
    stats = {i:{'value':[], 'area':[]} for i in range(1,21)}

    # for id in dataset.ids:
    qdar = tqdm.tqdm(dataset.ids, total=len(dataset.ids), ascii=True)
    for id in qdar:
        cam_dict = np.load(os.path.join(args.cam_out_dir, id + '.npy'), allow_pickle=True).item()
        cams = cam_dict['high_res']
        cams[cams<thres] = 0 #args.cam_eval_thres
        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=thres) #args.cam_eval_thres

        # for this particular thres, calc area and value sum. 
        value = np.reshape(cams,(cams.shape[0],-1)).sum(axis=1)
        area = copy.deepcopy(np.reshape(cams,(cams.shape[0],-1)))
        area[area>0] = 1
        area = area.sum(axis=1)

        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
        # save it to a class-keyed dictionary
        for i in range(len(keys)):
            if keys[i] == 0:
                continue
            stats[keys[i]]['value'].append(value[i])
            stats[keys[i]]['area'].append(area[i])

        cls_labels = np.argmax(cams, axis=0)
        if args.cam_eval_use_crf:
            img = np.asarray(imageio.imread(args.voc12_root+"JPEGImages/"+str(id)+'.jpg')) # load the original image 
            pred = imutils.crf_inference_label(img, cls_labels, n_labels=keys.shape[0]) # pass through CRF
            cls_labels = keys[pred]
        else:
            cls_labels = keys[cls_labels]
        preds.append(cls_labels.copy())
    
    # average stats.
    # import pdb;pdb.set_trace()
    for k,v in stats.items():
        vals = stats[k]['value']
        area = stats[k]['area']
        stats[k]['value'] = np.mean(vals)
        stats[k]['area'] = np.mean(area)

    confusion = calc_semantic_segmentation_confusion(preds, labels)

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator
    
    print({'iou': iou, 'miou': np.nanmean(iou)})
    
    # add iou value to stats
    for k,v in stats.items():
        stats[k]['iou'] = iou[k]
    return stats

def run(args):
    dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
    labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]
    
    stats = {}
    for thres in np.arange(0.01,0.15,0.02):
        stat = one_loop(dataset, labels, args, thres)
        stats[thres] = stat
    
    with open('stats.pkl','wb') as f:
        pickle.dump(stats,f)
    
