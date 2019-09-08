
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import voc12.dataloader
from misc import pyutils, torchutils, indexing
import importlib

def run(args):

    path_index = indexing.PathIndex(radius=10, default_size=(args.irn_crop_size // 4, args.irn_crop_size // 4))

    cam = getattr(importlib.import_module(args.cam_network), 'Net')()
    cam.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
    cam.eval()

    model = getattr(importlib.import_module(args.irn_network), 'Net')(cam)
    
    train_dataset = voc12.dataloader.VOC12AffinityDataset(args.train_list,
                                                          label_dir=args.ir_label_out_dir,
                                                          voc12_root=args.voc12_root,
                                                          indices_from=path_index.default_src_indices,
                                                          indices_to=path_index.default_dst_indices,
                                                          hor_flip=True,
                                                          crop_size=args.irn_crop_size,
                                                          crop_method="random",
                                                          rescale=(0.5, 1.5)
                                                          )
    train_data_loader = DataLoader(train_dataset, batch_size=args.irn_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    max_step = (len(train_dataset) // args.irn_batch_size) * args.irn_num_epoches

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': 1*args.irn_learning_rate, 'weight_decay': args.irn_weight_decay},
    ], lr=args.irn_learning_rate, weight_decay=args.irn_weight_decay, max_step=max_step)

    model = model.cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    for ep in range(args.irn_num_epoches):

        print('Epoch %d/%d' % (ep+1, args.irn_num_epoches))

        for iter, pack in enumerate(train_data_loader):

            img = pack['img'].cuda(non_blocking=True)
            label = pack['reduced_label'].cuda(non_blocking=True)
            # bg_pos_label = pack['aff_bg_pos_label'].cuda(non_blocking=True)
            # fg_pos_label = pack['aff_fg_pos_label'].cuda(non_blocking=True)
            # neg_label = pack['aff_neg_label'].cuda(non_blocking=True)

            pred = model(img, label)
            # TODO: masked pixel cross-entropy loss compute. 

            # avg_meter.add({'loss1': pos_aff_loss, 'loss2': neg_aff_loss, 'loss3': dp_fg_loss.item(), 'loss4': dp_bg_loss.item()})

            # total_loss = (pos_aff_loss + neg_aff_loss)/2 + (dp_fg_loss + dp_bg_loss)/2

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (optimizer.global_step-1)%100 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f %.4f %.4f %.4f' % (avg_meter.pop('loss1'), avg_meter.pop('loss2'), avg_meter.pop('loss3'), avg_meter.pop('loss4')),
                      'imps:%.1f' % ((iter+1) * args.irn_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)
        else:
            timer.reset_stage()

    torch.save(model.state_dict(), args.irn_weights_name)
    torch.cuda.empty_cache()
