import torch

#---------------------------------------------------------------------------#
#------------ Extract proposals from images given features -------------#
#---------------------------------------------------------------------------#

def extract_proposal_features(model, features, img_metas):
    assert model.with_bbox, 'Bbox head must be implemented.'

    proposal_list = model.rpn_head.simple_test_rpn(features, img_metas)
    return proposal_list


#---------------------------------------------------------------------------#
#-------------- Select Top-K Proposals ------------------#
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