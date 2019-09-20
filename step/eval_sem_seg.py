
import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
import imageio
import tqdm 

def run(args):
    dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
    labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]
    print(args.sem_seg_out_dir)
    preds = []
    qdar = tqdm.tqdm(dataset.ids,total=len(dataset.ids),ascii=True)
    cls_stats = np.zeros(21)
    for id in qdar:
        cls_labels = imageio.imread(os.path.join(args.sem_seg_out_dir, id + '.png')).astype(np.uint8)
        cls_labels[cls_labels == 255] = 0
        preds.append(cls_labels.copy())
        keys = np.unique(cls_labels)
        for k in keys:
            cls_stats[k] += 1

    # print("class stats: "+str(cls_stats))
    confusion = calc_semantic_segmentation_confusion(preds, labels)[:21, :21]

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    fp = 1. - gtj / denominator
    fn = 1. - resj / denominator
    iou = gtjresj / denominator

    print("fp and fn:")
    print("fp: "+str(np.round(fp,3)))
    print("fn: "+str(np.round(fn,3)))
    # print(fp[0], fn[0])
    print(np.mean(fp[1:]), np.mean(fn[1:]))

    print({'iou': iou, 'miou': np.nanmean(iou)})
