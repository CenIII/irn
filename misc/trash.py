def make_seg2clsbd_label_bak(seg_output,label,mask,args):
	'''
	Given seg_output of shape ([16, 21, 32, 32]), we make label for clsbd network. 
	return: clsbd_label
	'''
	# 0. detach off
	seg_output = seg_output.detach()
	mask = mask.detach()
	# 1. upsample
	w,h = seg_output.shape[-2:]
	seg_output = F.interpolate(seg_output, (w*4,h*4), mode='bilinear', align_corners=False) #[16, 21, 128, 128]
	mask = F.interpolate(mask, (w*4,h*4), mode='bilinear', align_corners=False) #[16, 21, 128, 128]
	# 2. softmax 
	probs = F.softmax(seg_output,dim=1)
	# 3. obtain & apply argmax mask
	preds = torch.argmax(probs,dim=1).unsqueeze(1)
	max_mask = probs.data.new(probs.shape).fill_(0.)
	max_mask = torch.scatter(max_mask,dim=1,index=preds,value=1.)
	probs_m = probs * max_mask
	# 4. select target classes
	probs_mtc = probs_m * label[:,:,None,None]
	# 5. thresholding fg, bg
	fg = probs_mtc[:,:-1]
	bg = probs_mtc[:,-1:]
	fg[fg<args.conf_fg_thres] = 0.
	bg[bg<args.conf_bg_thres] = 0.
	clsbd_label = torch.cat((fg,bg),dim=1)
	# 6. binarize 
	clsbd_label[clsbd_label>0] = 1.
	# 7. apply mask
	clsbd_label = clsbd_label * mask
	return clsbd_label, mask

def make_seg2clsbd_label_norm(seg_output,label,mask,args):
	'''
	Given seg_output of shape ([16, 21, 32, 32]), we make label for clsbd network. 
	return: clsbd_label
	'''
	# 0. detach off
	seg_output = seg_output.detach()
	mask = mask.detach()
	# 1. upsample
	w,h = seg_output.shape[-2:]
	seg_output = F.interpolate(seg_output, (w*4,h*4), mode='bilinear', align_corners=False) #[16, 21, 128, 128]
	mask = F.interpolate(mask, (w*4,h*4), mode='bilinear', align_corners=False) #[16, 21, 128, 128]
	
	norm_seg = seg_output / F.adaptive_max_pool2d(seg_output, (1, 1)) + 1e-5
	norm_seg = norm_seg * label[:,:,None,None]
	fg = norm_seg[:,:-1]
	bg = norm_seg[:,:-1].clone()
	fg[fg<args.conf_fg_thres] = 0.
	fg[fg>=args.conf_fg_thres] = 1.
	bg[bg<args.conf_bg_thres] = 0.
	bg[bg>=args.conf_bg_thres] = 1.
	bg = bg.sum(dim=1,keepdim=True)
	bg[bg>0] = 1
	bg = 1 - bg
	clsbd_label = torch.cat((fg,bg),dim=1)

	return clsbd_label, mask