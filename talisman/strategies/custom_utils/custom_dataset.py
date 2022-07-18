import os
import numpy as np
import json
from collections import Counter
from mmcv.parallel import collate, scatter
from mmdet.datasets import replace_ImageToTensor
from mmcv.ops import RoIPool
from mmdet.datasets.pipelines import Compose

#---------------------------------------------------------------------------#
#----------- Custom function to load dataset and return class --------------#
#---------------- wise object and image level statistics -------------------#
#---------------------------------------------------------------------------#

def get_class_statistics(dataset, indices):
  class_objects = {}   # define empty dict to hold class wise ground truths
  for i in range(len(dataset.CLASSES)):
    class_objects[i] = list()

  for i in indices:
    img_data, index = dataset[i]
    gt_labels = img_data['gt_labels'].data.numpy()
    for label in gt_labels:
        class_objects[label].append(index)
    
  #------------ print statistics -------------#
  print("Class".ljust(10), "No. of objects".ljust(3), "No. of images")
  print("-"*40)
  for key, val in class_objects.items():
    print(dataset.CLASSES[key].ljust(15), str(len(val)).ljust(15), len(set(val)))
  return class_objects

#---------------------------------------------------------------------------#
#--------------- Custom function to create Class Imbalance -----------------#
#---------------------------------------------------------------------------#

def create_custom_dataset(fullSet, all_indices, rare_class_budget, unrare_class_budget, imbalanced_classes, all_classes):

  labelled_budget = {}
  labelled_indices, unlabelled_indices = list(), list()
  exhausted_rare_classes = set()

  # initialize budget for rare and unrare class from the split_config file
  for i in range(len(fullSet.CLASSES)):
    if i in imbalanced_classes:
      labelled_budget[i] = rare_class_budget
    else:
      labelled_budget[i] = unrare_class_budget

  # iterate through whole dataset to select images class wise
  for i in all_indices:
    img_data, index = fullSet[i]
    #print(img_data)
    gt_labels = img_data['gt_labels'].data.numpy()
    
    # skip image if it does not contain classes with budget left
    if exhausted_rare_classes & set(gt_labels) or not (all_classes & set(gt_labels)):
      continue
    
    # else add image to the labelled pool and decrease budget class wise
    for label, no_of_objects in Counter(gt_labels).items():
        labelled_budget[label] -= no_of_objects # decrease budget

        if label in all_classes and labelled_budget[label] <= 0: # budget exhausted
          #print(fullSet.CLASSES[label]," class exhausted...")
          all_classes.remove(label)
          if label in imbalanced_classes:     # if rare class
            #print("added to rare class list")
            exhausted_rare_classes.add(label) # add to exhausted list of rare_classes
    
    labelled_indices.append(index)  # add image to labelled pool
    if not len(all_classes):        # if budget exceeded for all the classes, stop & return dataset
      #print("\nall class budget exhausted...")
      break


  # remove labelled indices from the full list
  labelled_indices = np.asarray(labelled_indices)
  unlabelled_indices = np.setdiff1d(all_indices, labelled_indices)

  # print dataset statistics
  stats = get_class_statistics(fullSet, labelled_indices)
  
  return labelled_indices, unlabelled_indices


#---------------------------------------------------------------------------#
#------- Prepare Validation set from labelled set -------#
#---------------------------------------------------------------------------#

def prepare_val_file(trn_dataset, indices, filename_07='trainval_07.txt', filename_12='trainval_12.txt', strat_dir='.'):
  trnval_07_file = open(os.path.join(strat_dir, filename_07), 'w')
  trnval_12_file = open(os.path.join(strat_dir,filename_12), 'w')
  for i, index in enumerate(indices):
    img_prefix = trn_dataset[index][0]['img_metas'].data['filename'].split('/')[-3]
    img_name = trn_dataset[index][0]['img_metas'].data['filename'].split('/')[-1].split('.')[0]
    if img_prefix == 'VOC2007':
      trnval_07_file.write(img_name + '\n')
    else:
      trnval_12_file.write(img_name + '\n')
  trnval_07_file.close()
  trnval_12_file.close()
  if os.path.getsize(trnval_07_file.name) and os.path.getsize(trnval_12_file.name):
    return [trnval_07_file.name, trnval_12_file.name]
  elif os.path.getsize(trnval_07_file.name):
    return trnval_07_file.name
  else:
    return trnval_12_file.name


#---------------------------------------------------------------------------#
#------------ Build the training dataset from the Config file --------------#
#---------------------------------------------------------------------------#

def build_dataset_with_indices(RepeatDataset):      # function to build dataset from config file and return with indices

    def __getitem__(self, index):
        data = RepeatDataset.__getitem__(self, index)
        return data, index

    return type(RepeatDataset.__name__, (RepeatDataset,), {
        '__getitem__': __getitem__,
    })

#---------------------------------------------------------------------------#
#------ Custom function to pass Training images through Test Pipeline ------#
#---------------------------------------------------------------------------#

def test_pipeline_images(model, imgs):
    """ Extract Convolutional features from the model backbone and Feature Pyramid Network neck.
    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image file names or loaded images.
    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the results directly.
    """
    print(imgs)
    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'
    
    imgs = data['img'][0] #length=1
    return imgs


#------- Functions for custom dataset computation for BDD100K -------------#

#---------------------------------------------------------------------------#
#-------- Custom function to create Class Imbalance for BDD dataset --------#
#---------------------------------------------------------------------------#
def create_custom_dataset_bdd(fullSet, all_indices, rare_class_budget, unrare_class_budget, imbalanced_classes, all_classes, attr_details, img_attribute_dict):

  labelled_budget = {}
  labelled_indices, unlabelled_indices = list(), list()
  exhausted_rare_classes = set()
  attr_class, attr_property, attr_value, attr_budget = attr_details
  # initialize budget for rare and unrare class from the split_config file
  for i in range(len(fullSet.CLASSES)):
    if i in imbalanced_classes:
      labelled_budget[i] = rare_class_budget
    else:
      labelled_budget[i] = unrare_class_budget
  
  # iterate through whole dataset to select images class wise
  for k,i in enumerate(all_indices):
    if not len(all_classes) and attr_budget <= 0:       # if budget exceeded for all the classes, stop & return dataset
      #print("\nall class budget exhausted...")
      break
    img_data, index = fullSet[i]
    gt_labels = img_data['gt_labels'].data.numpy()
    #break
    # skip image if it does not contain classes with budget left
    img_name = img_data['img_metas'].data['filename'].split('/')[-1]
    img_attr = img_attribute_dict[img_name][attr_property]
    if attr_class in gt_labels and img_attr == attr_value:
      if attr_budget > 0:
        # print("attr budget = ", attr_budget)
        # print("rare index -> ", index)
        labelled_indices.append(index)
        attr_budget -= sum(gt_labels == attr_class)
      continue
    if exhausted_rare_classes & set(gt_labels) or not (all_classes & set(gt_labels)):
      continue
    
    # else add image to the labelled pool and decrease budget class wise
    for label, no_of_objects in Counter(gt_labels).items():
        labelled_budget[label] -= no_of_objects # decrease budget
        
        if label in all_classes and labelled_budget[label] <= 0: # budget exhausted
          #print(fullSet.CLASSES[label]," class exhausted...")
          all_classes.remove(label)
          if label in imbalanced_classes:     # if rare class
            #print("added to rare class list")
            exhausted_rare_classes.add(label) # add to exhausted list of rare_classes
    
    labelled_indices.append(index)  # add image to labelled pool
  
  # remove labelled indices from the full list
  labelled_indices = np.asarray(labelled_indices)
  unlabelled_indices = np.setdiff1d(all_indices, labelled_indices)

  # print dataset statistics
  stats = get_class_statistics(fullSet, labelled_indices)
  
  return labelled_indices, unlabelled_indices

#---------------------------------------------------------------------------#
#---- Custom function to extract image wise attributes for BDD dataset -----#
#---------------------------------------------------------------------------#
def get_image_wise_attributes(json_file):
  # read det_train json file for image-attribute mapping
  rd_fl = open(json_file, 'r')
  str_data = rd_fl.read()
  image_data = json.loads(str_data)

  attribute_dict = {'weather': {'rainy': 0, 'snowy':0, 'clear':0, 'overcast':0, 'undefined':0, 'partly cloudy':0, 'foggy':0}, \
                    'scene': {'tunnel':0, 'residential':0, 'parking lot':0, 'undefined':0, 'city street':0, 'gas stations':0, 'highway':0}, \
                    'timeofday': {'daytime':0, 'night':0, 'dawn/dusk':0, 'undefined':0}}
  img_attribute_dict = {}
  img_names = list()
  for k,item in enumerate(image_data):
    w, d, s = item['attributes']['weather'], item['attributes']['timeofday'], item['attributes']['scene']
    attribute_dict['weather'][w] += 1
    attribute_dict['timeofday'][d] += 1
    attribute_dict['scene'][s] += 1
    img_attribute_dict[item['name']] = {'weather': item['attributes']['weather'], \
                                        'timeofday': item['attributes']['timeofday'], \
                                        'scene': item['attributes']['scene']}
  return attribute_dict, img_attribute_dict

#---------------------------------------------------------------------------#
#----- Custom function to get rare attribute statistics for BDD dataset ----#
#---------------------------------------------------------------------------#
def get_rare_attribute_statistics(dataset, indices, attr_details, img_attribute_dict, rare_class=True):  
  selected_rare_indices, no_of_rare_obj = list(), 0   # define empty list to hold image indices with rare attributes
  if rare_class:
    attr_class, attr_property, attr_value, attr_budget = attr_details
  else:
    attr_property, attr_value, rare_budget, unrare_budget = attr_details
  #print(len(indices))
  for i in indices:
    img_data, index = dataset[i]
    gt_labels = img_data['gt_labels'].data.numpy()
    img_name =  img_data['img_metas'].data['filename'].split('/')[-1]
    if img_attribute_dict[img_name][attr_property] == attr_value:
      if rare_class:
        if attr_class in gt_labels:
          for label in gt_labels:
            if label == attr_class:
              no_of_rare_obj += 1
          selected_rare_indices.append(index)
      else:
          no_of_rare_obj += 1
          selected_rare_indices.append(index)

  return selected_rare_indices, no_of_rare_obj

#---------------------------------------------------------------------------#
#------ Custom function to create attribute Imbalance for BDD dataset ------#
#---------------------------------------------------------------------------#
def create_dataset_with_only_attribute_imbalance_bdd(fullSet, all_indices, attr_imbalance_details, img_attribute_dict):
  rare_attr_type, rare_attr_value, rare_attr_budget, unrare_attr_budget = attr_imbalance_details
  labelled_indices, unlabelled_indices = list(), list()

  # iterate through whole dataset to select images class wise
  for k,i in enumerate(all_indices):
    if rare_attr_budget <= 0 and unrare_attr_budget <= 0: # if budget exceeded for both rare & unrare type, stop & return dataset
      #print("\nall type budget exhausted...")
      break
    img_data, index = fullSet[i]
    #break
    img_name = img_data['img_metas'].data['filename'].split('/')[-1]
    img_attr = img_attribute_dict[img_name][rare_attr_type]
    if img_attr == rare_attr_value:
      if rare_attr_budget > 0:
        rare_attr_budget -= 1
        labelled_indices.append(index)
    else:
      if unrare_attr_budget > 0:
        unrare_attr_budget -= 1
        labelled_indices.append(index)
  
  # remove labelled indices from the full list
  labelled_indices = np.asarray(labelled_indices)
  unlabelled_indices = np.setdiff1d(all_indices, labelled_indices)

  # print dataset statistics
  stats = get_class_statistics(fullSet, labelled_indices)
  
  return labelled_indices, unlabelled_indices

#---------------------------------------------------------------------------#
#---- Custom function to create rare attribute test file for BDD dataset ---#
#---------------------------------------------------------------------------#
def prepare_rare_test_file(json_file, attr_details, filename, rare_class_name=None):
  # read val_train json file for image-attribute mapping
  if rare_class_name:
    attr_class, attr_property, attr_value, _ = attr_details
  else:
    attr_property, attr_value, rare_budget, unrare_budget = attr_details
  # read validation json file
  rd_fl = open(json_file, 'r')
  str_data = rd_fl.read()
  image_data = json.loads(str_data)

  testfile = open(filename, "w")
  rare_test_img_count = 0
  for k, item in enumerate(image_data):
    img_name = item['name'].split('.')[0]
    if item['attributes'][attr_property] == attr_value:
      if rare_class_name:
        for label in item['labels']:
          if label['category'] == rare_class_name:
            # print(img_name, '->', label['category'], '+', item['attributes'][attr_property])
            testfile.write(img_name + '\n')
            rare_test_img_count += 1
            break
      else:
        # print(img_name, '->', item['attributes'][attr_property])
        testfile.write(img_name + '\n')
        rare_test_img_count += 1
  return rare_test_img_count
