import torch
from extract_features import get_RoI_features
from extract_proposals import extract_proposal_features


#---------------------------------------------------------------------------#
#-------- Define custom Uncertainty Score function : Score each image ------#
#-------- on the basis of Entropy and select the images with topmost -------#
#-------- Uncertainty scores for Next round of Uncertainty Sampling --------#
#---------------------------------------------------------------------------#

def get_uncertainty_scores(model, img_loader, no_of_imgs, imb_classes=None):

  uncertainty_scores = torch.zeros(no_of_imgs)
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

      # calculate class_entropies from the class probabilities
      # formula : entropy(p) = -[(p * logp) + {(1-p) * log(1-p)}] => (-p * logp) + {p * log(1-p)} - log(1-p)
      logp = torch.log2(batch_cls_scores)
      negp = torch.neg(batch_cls_scores)
      logOneMinusP = torch.log2(torch.add(negp, 1))
      batch_cls_scores = torch.add((negp * logp), torch.sub((batch_cls_scores * logOneMinusP),logOneMinusP))
      
      # split class_entropies as per no. of proposals in each image within batch
      num_proposals_per_img = tuple(len(p) for p in batch_proposals)
      batch_cls_scores = batch_cls_scores.split(num_proposals_per_img, 0)

      # for each image, take the max of class_entropies per proposal and aggregate over all proposals (average-max)
      for j, img_cls_scores in enumerate(batch_cls_scores):
        if imb_classes is not None:                 # use imbalanced class scores only for uncertainty score calculation
          imb_scores = torch.zeros(len(imb_classes))
          for k, imb_cls in enumerate(imb_classes):
            imb_scores[k] = torch.mean(img_cls_scores[:, imb_cls]) # average of each imb class over all proposals
          
          final_score = torch.max(imb_scores)                      # take max over all imb class averages
        else:                                       # use all class scores for uncertainty score calculation
          max_scores_per_proposal, _ = torch.max(img_cls_scores, dim=1) # take max of all class scores per proposal
          final_score = torch.mean(max_scores_per_proposal,dim=0)       # average over all proposals (avg-max implement)
        # store final uncertainty score for current image
        uncertainty_scores[indices[j]] = round(final_score.item(), 4)
      
  return uncertainty_scores