
import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
import imageio
from misc import torchutils, imutils
import tqdm
class_name = ['aeroplane', 'bicycle', 'bird', 'boat',
				'bottle', 'bus', 'car', 'cat', 'chair',
				'cow', 'diningtable', 'dog', 'horse',
				'motorbike', 'person', 'pottedplant',
				'sheep', 'sofa', 'train',
				'tvmonitor']
def run(args):
    dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
    labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]

    preds = []
    # for id in dataset.ids:
    qdar = tqdm.tqdm(dataset.ids, total=len(dataset.ids), ascii=True)
    for id in qdar:
        cam_dict = np.load(os.path.join(args.cam_out_dir, id + '.npy'), allow_pickle=True).item()
        cams = cam_dict['high_res']
        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
        cls_labels = np.argmax(cams, axis=0)
        img = np.asarray(imageio.imread(args.voc12_root+"JPEGImages/"+str(id)+'.jpg')) # load the original image 
        if args.cam_eval_use_crf:
            # img = np.asarray(imageio.imread(args.voc12_root+"JPEGImages/"+str(id)+'.jpg')) # load the original image 
            pred = imutils.crf_inference_label(img, cls_labels, n_labels=keys.shape[0]) # pass through CRF
            cls_labels = keys[pred]
        else:
            cls_labels = keys[cls_labels]
        preds.append(cls_labels.copy())

        imageio.imwrite(os.path.join(args.eval_out_dir, id + '.png'), img)
        for i in range(len(keys[1:])):
            k = keys[i+1]
            mask = np.zeros_like(cls_labels)
            mask[cls_labels==k] = 255.
            pred = cams[i+1]*mask
            imageio.imwrite(os.path.join(args.eval_out_dir, id + '_'+str(class_name[k-1])+'.png'), (pred).astype(np.uint8))

    confusion = calc_semantic_segmentation_confusion(preds, labels)

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator

    print({'iou': iou, 'miou': np.nanmean(iou)})
