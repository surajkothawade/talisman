import torch
import numpy as np
from tqdm import tqdm
from utils import _get_detector_cfg
from mmdet.apis import inference_detector


def get_gradients(cfg_file, inf_model, unlabelled_loader, device):
  print(cfg_file)
  model = _get_detector_cfg(cfg_file)
  model.backbone.init_cfg = None
  from mmdet.models import build_detector
  detector = build_detector(model)
  detector = detector.to(device)
  gradients = []
  for i, data_batch in enumerate(tqdm(unlabelled_loader)):     # for each batch
      # print("Computing gradients")
      # print(detector) 
      # sys.exit()
      detector.zero_grad()  
      # split the dataloader output into image_data and dataset indices
      img_data, img_indices = data_batch[0], data_batch[1].numpy()
      imgs, img_metas = img_data['img'].data[0].to(device), img_data['img_metas'].data[0]
      im_file_paths = []
      for im in img_metas:
        im_file_paths.append(im['filename'])

      #compute hypothesized labels for gradient 
      results = inference_detector(inf_model, im_file_paths)
      bboxes = []
      labels = []
      for cls_id in range(len(results[0])): #batch size is set to 1 for computing gradients, hence results[0] is used
        if len(results[0][cls_id])>0:
          for bbox in results[0][cls_id]:
            bboxes.append(bbox[:4].tolist())
            labels.append(cls_id)
      if len(labels) == 0: #use a vector of zero gradients if there are no pseudo detections
        grads = torch.zeros(1024,256) #change this to be the dimension of the gradients of the desired conv layer
        gradients.append(grads.cpu().numpy().flatten())
        continue

      # gt_bboxes = img_data['gt_bboxes'].data[0]
      # gt_labels = img_data['gt_labels'].data[0]

      # sys.exit()
      # for idx in range(len(gt_bboxes)):
      #   gt_bboxes[idx] = gt_bboxes[idx].to(device) 
      #   gt_labels[idx] = gt_labels[idx].to(device)
      bboxes = [torch.Tensor(bboxes).to(device)]
      labels = [torch.Tensor(labels).long().to(device)]
      
      # print(bboxes, labels)
      # print(gt_bboxes, gt_labels)

      losses = detector.forward(imgs, img_metas, gt_bboxes=bboxes, gt_labels=labels, return_loss=True)
      assert isinstance(losses, dict)
      loss, _ = detector._parse_losses(losses)
      loss.requires_grad_(True)
      assert float(loss.item()) > 0
      loss.backward()
      # grads = detector.backbone.layer4[2].conv3.weight.grad
      grads = detector.backbone.layer3[5].conv3.weight.grad
      gradients.append(grads.cpu().numpy().flatten())
      # print("==================GRADIENTS========================")
      # print(grads.shape)
      # print(grads)
      # print("backward pass complete")
  gradients = np.stack(gradients,0)
  return gradients
