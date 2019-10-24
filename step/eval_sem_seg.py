
import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
import imageio
import tqdm 

def load_img_name_list(dataset_path):

    img_name_list = np.loadtxt(dataset_path, dtype=np.int32)

    return img_name_list
def decode_int_filename(int_filename):
    s = str(int(int_filename))
    return s[:4] + '_' + s[4:]

def run(args):
    dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
    labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]
    print(args.sem_seg_out_dir)
    preds = []
    qdar = tqdm.tqdm(dataset.ids,total=len(dataset.ids),ascii=True)
    for id in qdar:
        cls_labels = imageio.imread(os.path.join(args.sem_seg_out_dir, id + '.png')).astype(np.uint8)
        cls_labels[cls_labels == 255] = 0
        preds.append(cls_labels.copy())
        keys = np.unique(cls_labels)
    confusion = calc_semantic_segmentation_confusion(preds, labels)[:21, :21] #[labels[ind] for ind in ind_list]

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
