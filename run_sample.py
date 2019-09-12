import argparse
import os

from misc import pyutils

def join_path(args):
    args.exp_root = os.path.join('exp', args.exp_root)
    args.log_name = os.path.join(args.exp_root, args.log_name)
    args.cam_weights_name = os.path.join(args.exp_root, args.cam_weights_name)
    args.irn_weights_name = os.path.join(args.exp_root, args.irn_weights_name)
    args.cam_out_dir = os.path.join(args.exp_root, args.cam_out_dir)
    args.vis_out_dir = os.path.join(args.exp_root, args.vis_out_dir)
    args.ir_label_out_dir = os.path.join(args.exp_root, args.ir_label_out_dir)
    args.sem_seg_out_dir = os.path.join(args.exp_root, args.sem_seg_out_dir)
    args.ins_seg_out_dir = os.path.join(args.exp_root, args.ins_seg_out_dir)

    os.makedirs('./exp', exist_ok=True)
    os.makedirs(args.exp_root, exist_ok=True)
    os.makedirs(os.path.join(args.exp_root,"sess"), exist_ok=True)
    os.makedirs(os.path.join(args.exp_root,"result"), exist_ok=True)
    os.makedirs(args.cam_out_dir, exist_ok=True)
    os.makedirs(args.vis_out_dir, exist_ok=True)
    os.makedirs(args.ir_label_out_dir, exist_ok=True)
    os.makedirs(args.sem_seg_out_dir, exist_ok=True)
    os.makedirs(args.ins_seg_out_dir, exist_ok=True)
    return args

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument("--voc12_root", required=True, type=str)

    # Dataset
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--infer_list", default="voc12/train.txt", type=str)
    parser.add_argument("--chainer_eval_set", default="train", type=str)
    parser.add_argument("--quick_list", default="voc12/quick.txt", type=str)

    # Class Activation Map
    parser.add_argument("--cam_network", default="net.resnet50_cam", type=str)
    parser.add_argument("--cam_crop_size", default=512, type=int)
    parser.add_argument("--cam_batch_size", required=True, type=int)
    parser.add_argument("--cam_num_epoches", default=5, type=int)
    parser.add_argument("--cam_learning_rate", required=True, type=float)
    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--cam_eval_thres", default=0.15, type=float)
    parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0))
    parser.add_argument("--cam_visualize_train", default=False)

    # Mining Inter-pixel Relations
    parser.add_argument("--conf_fg_thres", default=0.30, type=float)
    parser.add_argument("--conf_bg_thres", default=0.05, type=float)

    # Inter-pixel Relation Network (IRNet)
    parser.add_argument("--irn_network", default="net.resnet50_clsbd", type=str)
    parser.add_argument("--irn_crop_size", default=512, type=int)
    parser.add_argument("--irn_batch_size", required=True, type=int)
    parser.add_argument("--irn_num_epoches", default=3, type=int)
    parser.add_argument("--irn_learning_rate", required=True, type=float)
    parser.add_argument("--irn_weight_decay", default=1e-4, type=float)
    parser.add_argument("--quick_make_sem", default=False)
    # Random Walk Params
    parser.add_argument("--beta", default=10)
    parser.add_argument("--exp_times", default=8)
    parser.add_argument("--ins_seg_bg_thres", default=0.25)
    parser.add_argument("--sem_seg_bg_thres", default=0.25)

    # Output Path
    parser.add_argument("--exp_root", default="exp0", type=str)
    parser.add_argument("--log_name", default="sample_train_eval", type=str)
    parser.add_argument("--cam_weights_name", default="sess/res50_cam.pth", type=str)
    parser.add_argument("--irn_weights_name", default="sess/res50_irn_clsbd.pth", type=str)
    parser.add_argument("--cam_out_dir", default="result/cam", type=str)
    parser.add_argument("--vis_out_dir", default="result/vis_clsbd", type=str)
    parser.add_argument("--ir_label_out_dir", default="result/ir_label", type=str)
    parser.add_argument("--sem_seg_out_dir", default="result/sem_seg_clsbd", type=str)
    parser.add_argument("--ins_seg_out_dir", default="result/ins_seg", type=str)

    # Step
    parser.add_argument("--train_cam_pass", default=True)
    parser.add_argument("--make_cam_pass", default=True)
    parser.add_argument("--eval_cam_pass", default=True)
    parser.add_argument("--cam_to_ir_label_pass", default=False)
    parser.add_argument("--train_irn_pass", default=True)
    parser.add_argument("--make_ins_seg_pass", default=False)
    parser.add_argument("--eval_ins_seg_pass", default=False)
    parser.add_argument("--make_sem_seg_pass", default=True)
    parser.add_argument("--eval_sem_seg_pass", default=True)

    args = parser.parse_args()

    args = join_path(args)

    pyutils.Logger(args.log_name + '.log')
    print(vars(args))

    if args.train_cam_pass is True:
        import step.train_cam

        timer = pyutils.Timer('step.train_cam:')
        step.train_cam.run(args)

    if args.make_cam_pass is True:
        import step.make_cam

        timer = pyutils.Timer('step.make_cam:')
        step.make_cam.run(args)

    if args.eval_cam_pass is True:
        import step.eval_cam

        timer = pyutils.Timer('step.eval_cam:')
        step.eval_cam.run(args)

    if args.cam_to_ir_label_pass is True:
        import step.cam_to_ir_label

        timer = pyutils.Timer('step.cam_to_ir_label:')
        step.cam_to_ir_label.run(args)

    if args.train_irn_pass is True:
        import step.train_clsbd

        timer = pyutils.Timer('step.train_clsbd:')
        step.train_clsbd.run(args)

    if args.make_ins_seg_pass is True:
        import step.make_ins_seg_labels

        timer = pyutils.Timer('step.make_ins_seg_labels:')
        step.make_ins_seg_labels.run(args)

    if args.eval_ins_seg_pass is True:
        import step.eval_ins_seg

        timer = pyutils.Timer('step.eval_ins_seg:')
        step.eval_ins_seg.run(args)

    if args.make_sem_seg_pass is True:
        import step.make_sem_seg_labels

        timer = pyutils.Timer('step.make_sem_seg_labels:')
        step.make_sem_seg_labels.run(args)

    if args.eval_sem_seg_pass is True:
        import step.eval_sem_seg

        timer = pyutils.Timer('step.eval_sem_seg:')
        step.eval_sem_seg.run(args)

