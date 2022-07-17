import torch

#---------------------------------------------------------------------------#
#------ Custom function to find diverse data points from unlabelled --------#
#------ set that are furthest from the current labelled dataset ------------#
#---------------------------------------------------------------------------#
def furthest_first(model, unlabeled_embeddings, labeled_embeddings, n):
  device = next(model.parameters()).device
  unlabeled_embeddings = torch.flatten(torch.tensor(unlabeled_embeddings).to(device), start_dim=1)
  print('Unlabelled embeddings shape: ', unlabeled_embeddings.shape)
  labeled_embeddings = torch.flatten(torch.tensor(labeled_embeddings).to(device), start_dim=1)
  print('Labelled embeddings shape: ', labeled_embeddings.shape)
  
  m = unlabeled_embeddings.shape[0]
  if labeled_embeddings.shape[0] == 0:
      min_dist = torch.tile(float("inf"), m)
  else:
      dist_ctr = torch.cdist(unlabeled_embeddings, labeled_embeddings, p=2)
      min_dist = torch.min(dist_ctr, dim=1)[0]
          
  idxs = []
  
  for i in range(n):
      idx = torch.argmax(min_dist)
      idxs.append(idx.item())
      dist_new_ctr = torch.cdist(unlabeled_embeddings, unlabeled_embeddings[[idx],:])
      min_dist = torch.minimum(min_dist, dist_new_ctr[:,0])
          
  return idxs