import torch
from tqdm import tqdm
from extract_proposals import extract_proposal_features
from extract_features import get_RoI_features

#---------------------------------------------------------------------------#
#------ Define Least Margin Score function : Score each image on the -------#
#------ basis of difference in top-2 max class confidence and select -------#
#------ the top images with least margin scores for next round training ----#
#---------------------------------------------------------------------------#
def get_margin_scores(model, img_loader, no_of_imgs, imb_classes=None):

  margin_scores = torch.ones(no_of_imgs)
  device = next(model.parameters()).device  # model device

  if imb_classes is not None:
    print('using imbalanced classes ', imb_classes)

  for i, data_batch in enumerate(tqdm(img_loader)):     # for each batch
            
      # split the dataloader output into image_data and dataset indices
      img_data, indices = data_batch[0], data_batch[1].numpy()
      
      imgs, img_metas = img_data['img'].data[0].to(device=device), img_data['img_metas'].data[0]
      
      # extract image features from backbone + FPN neck
      with torch.no_grad():
          features = model.extract_feat(imgs)
      
      # get batch proposals from RPN Head and extract class scores from RoI Head
      batch_proposals = extract_proposal_features(model, features, img_metas)
      batch_cls_scores = get_RoI_features(model, features, batch_proposals, only_cls_scores=True)
      
      # normalize class_scores for each image to range between (0,1) which indicates
      # probability whether an object of that class has a bounding box centered there
      batch_cls_scores = batch_cls_scores.softmax(-1)

      # split class_entropies as per no. of proposals in each image within batch
      num_proposals_per_img = tuple(len(p) for p in batch_proposals)
      batch_cls_scores = batch_cls_scores.split(num_proposals_per_img, 0)

      # for each image, take the max of class_entropies per proposal and aggregate over all proposals (average-max)
      for j, img_cls_scores in enumerate(batch_cls_scores):
        #print(img_cls_scores[0])
        sorted_scores_per_proposal, _ = torch.sort(img_cls_scores, descending=True) # sort all class scores per proposal
        sorted_scores_per_proposal = sorted_scores_per_proposal[:, 0] - sorted_scores_per_proposal[:, 1] # take diff of top 2 max scores per proposal
        final_score = torch.mean(sorted_scores_per_proposal, dim=0)       # average over all proposals (avg-maxdiff implement)
        # store final margin score for current image
        margin_scores[indices[j]] = final_score.item()
  
  return margin_scores