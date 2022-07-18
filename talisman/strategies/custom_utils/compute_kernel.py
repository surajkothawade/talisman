import torch
import numpy as np
import math
from tqdm import tqdm

#---------------------------------------------------------------------------#
#---------------------- Custom function to L2 Normalize --------------------#
#---------------------------------------------------------------------------#

def l2_normalize(a, axis=-1, order=2):
    #L2 normalization that works for any arbitary axes
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


#---------------------------------------------------------------------------#
#----------- Query-Query kernel computation ------------#
#---------------------------------------------------------------------------#

def compute_queryQuery_kernel(query_dataset_feat):
    query_query_sim = []
    for i in range(len(query_dataset_feat)):
        query_row_sim = []
        for j in range(len(query_dataset_feat)):
            query_feat_i = query_dataset_feat[i] #(num_proposals, num_features)
            query_feat_j = query_dataset_feat[j]
            query_feat_i = l2_normalize(query_feat_i) 
            query_feat_j = l2_normalize(query_feat_j)
            dotp = np.tensordot(query_feat_i, query_feat_j, axes=([1],[1])) #compute the dot product along the feature dimension, i.e between every GT bbox of rare class in the query image
            max_match_queryGt_queryGt = np.amax(dotp, axis=(0,1)) #get the max from (num_proposals in query i, num_proposals in query j)
            query_row_sim.append(max_match_queryGt_queryGt)
        query_query_sim.append(query_row_sim)
    query_query_sim = np.array(query_query_sim)
    print("final query image kernel shape: ", query_query_sim.shape)
    return query_query_sim

#---------------------------------------------------------------------------#
#----------------- Query-Image kernel computation ------------#
#---------------------------------------------------------------------------#

def compute_queryImage_kernel(query_dataset_feat, unlabeled_dataset_feat):
    query_image_sim = []
    unlabeled_feat_norm = l2_normalize(unlabeled_dataset_feat) #l2-normalize the unlabeled feature vector along the feature dimension (batch_size, num_proposals, num_features)
    for i in range(len(query_dataset_feat)):
        query_feat = np.expand_dims(query_dataset_feat[i], axis=0)
        query_feat_norm = l2_normalize(query_feat) #l2-normalize the query feature vector along the feature dimension
        #print(query_feat_norm.shape)
        #print(unlabeled_feat_norm.shape)
        dotp = np.tensordot(query_feat_norm, unlabeled_feat_norm, axes=([2],[2])) #compute the dot product along the feature dimension, i.e between every GT bbox of rare class in the query image with all proposals from all images in the unlabeled set
        #print(dotp.shape)
        max_match_queryGt_proposal = np.amax(dotp, axis=(1,3)) #find the gt-proposal pair with highest similarity score for each image
        query_image_sim.append(max_match_queryGt_proposal)
    query_image_sim = np.vstack(tuple(query_image_sim))
    print("final query image kernel shape: ", query_image_sim.shape)
    return query_image_sim

#---------------------------------------------------------------------------#
#----------------------Image-Image kernel computation -------------#
#---------------------------------------------------------------------------#

def compute_imageImage_kernel(unlabeled_dataset_feat, device="cpu", batch_size=100):
    print("Computing image image kernel")
    image_image_sim = []
    unlabeled_dataset_feat = l2_normalize(unlabeled_dataset_feat) #l2-normalize the unlabeled feature vector along the feature dimension
    unlabeled_dataset_feat = torch.Tensor(unlabeled_dataset_feat).to(device)
    print("unlabeled_dataset_feat.shape: ", unlabeled_dataset_feat.shape, type(unlabeled_dataset_feat))
    unlabeled_data_size = unlabeled_dataset_feat.shape[0]
    for i,_ in enumerate(tqdm(range(math.ceil(unlabeled_data_size/batch_size)))): #batch through the unlabeled dataset to compute the similarity matrix
        start_ind = i*batch_size
        end_ind = start_ind + batch_size
        if(end_ind > unlabeled_data_size):
            end_ind = unlabeled_data_size
        unlabeled_feat_batch = unlabeled_dataset_feat[start_ind:end_ind,:,:]
        if(device.startswith("cuda")):
          dotp = torch.tensordot(unlabeled_feat_batch, unlabeled_dataset_feat, dims=([2],[2]))
          max_match_unlabeledProposal_proposal = torch.amax(dotp, axis=(1,3)).cpu().numpy()
        else:
          dotp = np.tensordot(unlabeled_feat_batch, unlabeled_dataset_feat, axes=([2],[2])) #compute the dot product along the feature dimension, i.e between every proposal in an unlabeled image with all proposals from all images in the unlabeled set
          max_match_unlabeledProposal_proposal = np.amax(dotp, axis=(1,3)) #find the proposal-proposal pair with highest similarity score for each image
        image_image_sim.append(max_match_unlabeledProposal_proposal)
    image_image_sim = np.vstack(tuple(image_image_sim))
    print(image_image_sim.shape)
    return image_image_sim