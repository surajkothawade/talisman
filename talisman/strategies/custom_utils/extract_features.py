'''Functions to extract features from regions of interest or from the complete image.Different functions can be used depending on if the image exists in the query set and has a ground truth region of interest or a set of proposals for an image from the unlabeled set.'''

import numpy as np
import torch
from mmdet.core import bbox2roi
from tqdm import tqdm

#---------------------------------------------------------------------------#
#-------------- Extract proposals from images given features ---------------#
#---------------------------------------------------------------------------#

def extract_proposal_features(model, features, img_metas):
    assert model.with_bbox, 'Bbox head must be implemented.'

    proposal_list = model.rpn_head.simple_test_rpn(features, img_metas)
    return proposal_list


#---------------------------------------------------------------------------#
#------------------------ Select Top-K Proposals ---------------------------#
#---------------------------------------------------------------------------#

def select_top_k_proposals(fg_cls_scores, fg_classes_with_max_score, fg_classes, proposal_budget):
  # get the indices in order which sorts the foreground class proposals scores in descending order
  max_score_order = torch.argsort(fg_cls_scores, descending=True).tolist()
  
  selected_prop_indices = list()
  # loop through until proposal budget is exhausted
  while proposal_budget:
    cls_budget, per_cls_budget, next_round_max_score_order =  dict(), (proposal_budget // len(fg_classes)) + 1, list()
    # assign budget to each foreground class
    for cls in fg_classes:
      cls_budget[cls.item()] = per_cls_budget
    
    # loop through the ordered list
    for idx in max_score_order:
      curr_class = fg_classes_with_max_score[idx].item()
      if cls_budget[curr_class]: # if budget permits
        selected_prop_indices.append(idx)   # add index to selection list
        cls_budget[curr_class] -= 1         # reduce class budget
        proposal_budget -= 1                # reduce proposal budget
        if not proposal_budget:             # stop if proposal budget exhausted
          break
      else:
        next_round_max_score_order.append(idx)
    # limit the order_list to indices not chosen in current iteration
    max_score_order = next_round_max_score_order
    
  return selected_prop_indices

#---------------------------------------------------------------------------#
#----------------------- Extract resized features from ---------------------#
#-------------------- different layers after RoI Pooling -------------------#
#---------------------------------------------------------------------------#

def get_RoI_features(model, features, proposals, with_shared_fcs=False, only_cls_scores=False):
  """ Extract features from either the RoI pooling layers or shared Fully-connected layers
      or directly return the class_scores from the final class predictor itself.
    Args:
        model (nn.Module): The loaded detector.
        features  (tuple[Tensor]): Features from the upstream network, each is a 4D-tensor. 
        proposals (list[Tensor]): Either predicted proposals from RPN layers (unlabelled images)
                                  or transformed proposals from ground truth bounding boxes (query set).
        with_shared_fcs (Bool): if True, return the features from shared FC layer; default is False
        only_cls_scores (Bool): if True, return the class_scores from the final predictor; default is False
    Returns:
        (List[Tensor]) : If 'only_cls_scores' flag is set, class_scores from the final predictor for each 
                         proposal will be returned, otherwise return the feature maps after flattening out.
  """

  device = next(model.parameters()).device      # model device
  rois = bbox2roi(proposals).to(device=device)  # convert proposals to Region of Interests
  bbox_feats = model.roi_head.bbox_roi_extractor(
            features[:model.roi_head.bbox_roi_extractor.num_inputs], rois)
  
  if model.roi_head.with_shared_head:
            bbox_feats = model.roi_head.shared_head(bbox_feats)
  
  #print("Features shape from RoI Pooling Layer: ",bbox_feats.shape) # [no_of_proposals, 256, 7, 7]
  x = bbox_feats.flatten(1)       # flatten the RoI Pooling features

  if with_shared_fcs or only_cls_scores: # extract flattened features from shared FC layers
      for fc in model.roi_head.bbox_head.shared_fcs:
          x = model.roi_head.bbox_head.relu(fc(x))

      if only_cls_scores:           # if cls_scores flag is set
        cls_scores = model.roi_head.bbox_head.fc_cls(x) if model.roi_head.bbox_head.with_cls else None
        return cls_scores           # return class scores from the final class predictors
      # else return output from the shared_fc layers
      return x
  # else return features from the RoI pooling layer
  return bbox_feats



#---------------------------------------------------------------------------#
#--------------- Extract RoI features from Unlabelled set ------------------#
#---------------------------------------------------------------------------#

def get_unlabelled_RoI_features(model, unlabelled_loader, feature_type):

  device = next(model.parameters()).device  # model device
  unlabelled_indices = list()
  unlabeled_features = []
  if(feature_type == "fc"):
    fc_features = True
  for i, data_batch in enumerate(tqdm(unlabelled_loader)):     # for each batch
            
      # split the dataloader output into image_data and dataset indices
      img_data, indices = data_batch[0], data_batch[1].numpy()
      
      imgs, img_metas = img_data['img'].data[0].to(device=device), img_data['img_metas'].data[0]
      
      # extract image features from backbone + FPN neck
      with torch.no_grad():
        features = model.extract_feat(imgs)
      
      # get batch proposals from RPN Head and extract class scores from RoI Head
      batch_proposals = extract_proposal_features(model, features, img_metas)
      batch_roi_features = get_RoI_features(model, features, batch_proposals, with_shared_fcs=fc_features)
      
      num_proposals_per_img = tuple(len(p) for p in batch_proposals)
      batch_roi_features = batch_roi_features.split(num_proposals_per_img, 0)
            
      for j, img_roi_features in enumerate(batch_roi_features):
#         print(indices[j], img_roi_features.shape)
        unlabelled_indices.append(indices[j]) # add image index to list
        xf = img_roi_features.detach().cpu().numpy()
        unlabeled_features.append(xf)

  unlabeled_features = np.stack(unlabeled_features, axis=0)
  return unlabeled_features, unlabelled_indices


#---------------------------------------------------------------------------#
#----------------------- Extract RoI features from  ------------------------#
#----------------- Unlabelled set with Top-K Proposals ---------------------#
#---------------------------------------------------------------------------#
def get_unlabelled_top_k_RoI_features(model, unlabelled_loader, proposal_budget, feature_type):

  device = next(model.parameters()).device  # model device
  unlabelled_indices = list()
  unlabelled_roi_features = list()

  if(feature_type == "fc"):
    fc_features = True

  for i, data_batch in enumerate(tqdm(unlabelled_loader)):     # for each batch
            
      # split the dataloader output into image_data and dataset indices
      img_data, img_indices = data_batch[0], data_batch[1].numpy()
      
      imgs, img_metas = img_data['img'].data[0].to(device=device), img_data['img_metas'].data[0]
      
      # extract image features from backbone + FPN neck
      with torch.no_grad():
          features = model.extract_feat(imgs)
      
      # get batch proposals from RPN Head and extract class scores from RoI Head
      batch_proposals = extract_proposal_features(model, features, img_metas)
      batch_roi_features = get_RoI_features(model, features, batch_proposals, with_shared_fcs=True)
      batch_cls_scores = get_RoI_features(model, features, batch_proposals, only_cls_scores=True)

      # normalize class_scores for each image to range between (0,1) which indicates
      # probability whether an object of that class has a bounding box centered there
      batch_cls_scores = batch_cls_scores.softmax(-1)
      
      # split features and cls_scores
      num_proposals_per_img = tuple(len(p) for p in batch_proposals)
      batch_cls_scores = batch_cls_scores.split(num_proposals_per_img, 0)
      batch_roi_features = batch_roi_features.split(num_proposals_per_img, 0)

      # for each image, select the top-k proposals where k = proposal_budget
      for j, img_cls_scores in enumerate(batch_cls_scores):
          img_roi_features = batch_roi_features[j]
          max_score_per_proposal, max_score_classes = torch.max(img_cls_scores, dim=1) # take max of all class scores per proposal
          classes, indices, counts = torch.unique(max_score_classes, return_inverse=True, return_counts=True)
          
          bg_class_index, bg_count, num_proposals  = len(classes) - 1, counts[-1], len(indices)
          fg_indices = indices != bg_class_index
          #print(classes, indices, counts)
          fg_img_cls_scores = max_score_per_proposal[fg_indices]
          fg_classes_with_max_score = max_score_classes[fg_indices]
          fg_img_roi_features = img_roi_features[fg_indices]
          #print(fg_img_roi_features.shape)
          
          if bg_count > num_proposals - proposal_budget: # no. of foreground proposals < proposal_budget
            #print("augment some background imgs")
            bg_indices = indices == bg_class_index
            bg_img_roi_features = img_roi_features[bg_indices][:bg_count - num_proposals + proposal_budget]
            selected_roi_features = torch.cat((fg_img_roi_features, bg_img_roi_features)).detach().cpu().numpy()
            del bg_indices, bg_img_roi_features
          elif bg_count == num_proposals - proposal_budget: # no. of foreground proposals = proposal_budget
            #print("no need to augment or select")
            selected_roi_features = fg_img_roi_features.detach().cpu().numpy()
          else:                                             # no. of foreground proposals > proposal_budget
            #print("select from foreground imgs")
            top_k_indices = select_top_k_proposals(fg_img_cls_scores, fg_classes_with_max_score, classes[:-1], proposal_budget)
            #print(fg_classes_with_max_score[top_k_indices])
            selected_roi_features = fg_img_roi_features[top_k_indices].detach().cpu().numpy()
          
          # append to unlebelled_roi_features list
          unlabelled_roi_features.append(selected_roi_features)
          unlabelled_indices.append(img_indices[j]) # add image index to list
          # free up gpu_memory
          del max_score_per_proposal, max_score_classes, classes, indices, counts, bg_class_index, bg_count, num_proposals,fg_indices, fg_img_cls_scores, fg_classes_with_max_score, fg_img_roi_features
          
  unlabelled_features = np.stack(unlabelled_roi_features, axis=0)
  return unlabelled_features, unlabelled_indices

#---------------------------------------------------------------------------#
#-------------------- Extract RoI features from Query set ------------------#
#---------------------------------------------------------------------------#

def get_query_RoI_features(model, query_loader, imbalanced_classes, feature_type):

  device = next(model.parameters()).device  # model device
  query_indices = list()
  query_features = []
  if(feature_type == "fc"):
    fc_features = True
  for i, data_batch in enumerate(tqdm(query_loader)):     # for each batch
            
      # split the dataloader output into image_data and dataset indices
      img_data, indices = data_batch[0], data_batch[1].numpy()
      
      imgs, img_metas = img_data['img'].data[0].to(device=device), img_data['img_metas'].data[0]
      batch_gt_bboxes = img_data['gt_bboxes'].data[0]          # extract gt_bboxes from data batch
      batch_gt_labels = img_data['gt_labels'].data[0]          # extract gt_labels from data batch

      gt_bboxes, gt_labels = list(), list()
      # filter only the imbalanced class bboxes and labels
      for img_gt_bboxes, img_gt_labels in zip(batch_gt_bboxes, batch_gt_labels):
        #print(img_gt_bboxes, img_gt_labels)
        imb_cls_indices = torch.zeros(len(img_gt_labels), dtype=torch.bool)
        for imb_class in imbalanced_classes:
          imb_cls_indices = (imb_cls_indices | torch.eq(img_gt_labels, imb_class))
        
        #print('rare class:',img_gt_labels[imb_cls_indices], img_gt_bboxes[imb_cls_indices])
        gt_bboxes.append(img_gt_bboxes[imb_cls_indices])
        gt_labels.append(img_gt_labels[imb_cls_indices])
      
      num_gts_per_img = tuple(len(p) for p in gt_bboxes) # store how many bboxes per img
      #print(num_gts_per_img)
      #print(gt_bboxes, gt_labels)
      
      gt_bboxes = torch.cat(gt_bboxes)                   # stack all bboxes across batch of imgs
      gt_labels = torch.cat(gt_labels)                   # stack all labels across batch of imgs
      #print(gt_bboxes, gt_labels)
      
      # append confidence score of 1.0 to each gt_bboxes
      batch_proposals = torch.cat((gt_bboxes, torch.ones(gt_bboxes.shape[0], 1)), 1)
      # return batch proposals to original shape as were in batch
      batch_proposals =  batch_proposals.split(num_gts_per_img, 0)
      
      # extract image features from backbone + FPN neck
      with torch.no_grad():
          features = model.extract_feat(imgs)
      
      batch_roi_features = get_RoI_features(model, features, batch_proposals, with_shared_fcs=fc_features)
      batch_roi_features = batch_roi_features.split(num_gts_per_img, 0)
      
      for j, img_roi_features in enumerate(batch_roi_features):
        #print(indices[j], img_roi_features.shape)
        query_indices.append(indices[j]) # add image index to list
        xf = img_roi_features.detach().cpu().numpy()
        query_features.append(xf)
      
#   query_features = np.stack(query_features, axis=0)
  return query_features, query_indices


#---------------------------------------------------------------------------#
#------- Custom function to extract global features from backbone ----------#
#---------------------------------------------------------------------------#
def extract_global_descriptor(model, img_loader, no_of_imgs=None):
  device = next(model.parameters()).device  # model device
  img_features = list()
  img_indices = list()
  for i, data_batch in enumerate(tqdm(img_loader)):     # for each batch
              
        # split the dataloader output into image_data and dataset indices
        img_data, indices = data_batch[0], data_batch[1].numpy()
        
        imgs, img_metas = img_data['img'].data[0].to(device=device), img_data['img_metas'].data[0]
        
        # extract image features from backbone
        with torch.no_grad():
            batch_features = model.backbone(imgs) # extract 6 feature vectors of varying dims
            for j, img_feature in enumerate(batch_features[-1]): # take the last feature vector of 2048 * 18 * 32
              # print(img_feature.shape)
              img_feature, _ = torch.max(img_feature, dim=0)     # take max of all 2048 channel dimention
              # print(img_feature.shape)
              img_features.append(img_feature)  # add feature to list
              img_indices.append(indices[j])    # add image index to list
  return img_features, img_indices


#---------------------------------------------------------------------------#
#------ Custom function to extract the smallest backbone + fpn features  ---#
#------ from passed dataset of dim size * 18 * 32 along with indices -------#
#---------------------------------------------------------------------------#
def extract_embeddings(model, img_loader):
  device = next(model.parameters()).device    # model device      
  embeddings = []
  embedding_indices = list()
  for i, data_batch in enumerate(tqdm(img_loader)): # for each batch
        # split the dataloader output into image_data and dataset indices
        img_data, indices = data_batch[0], data_batch[1].numpy()
        # print(indices)
        imgs, img_metas = img_data['img'].data[0].to(device=device), img_data['img_metas'].data[0]
        
        # extract image features from backbone + fpn
        with torch.no_grad():
            batch_features, _ = torch.max(model.extract_feat(imgs)[-2],dim=1)
            # print(features.shape)
        for j, img_features in enumerate(batch_features):
            xf = img_features.detach().cpu().numpy()
            embeddings.append(xf)
            embedding_indices.append(indices[j])
  return np.array(embedding_indices), np.array(embeddings)