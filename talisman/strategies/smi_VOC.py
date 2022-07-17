# import numpy
import numpy as np
import os
import sys
import gc

# import submodlib
import submodlib
from talisman.utils.compute_kernel import compute_imageImage_kernel, compute_queryImage_kernel, compute_queryQuery_kernel
from talisman.utils.custom_dataset import build_dataset_with_indices, create_custom_dataset, get_class_statistics, prepare_val_file
from talisman.utils.extract_features import get_query_RoI_features, get_unlabelled_RoI_features, get_unlabelled_top_k_RoI_features
from talisman.utils.utils import execute

# Check Pytorch installation
import torch, torchvision
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
print(torch.__version__, torch.cuda.is_available())

# Check MMDetection installation
import mmdet
print(mmdet.__version__)

# Check mmcv installation
import mmcv
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())

# import other modules
import copy

# import mmcv functionalities
from mmcv import Config
from mmdet.apis import init_detector
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.datasets.dataset_wrappers import RepeatDataset


#---------------------------------------------------------------------------#
#------------------ initialize training parameters -------------------------#
#---------------------------------------------------------------------------#
budget = 200    # set Active Learning Budget
no_of_rounds= 8 # No. of Rounds to run
max_epochs=150  # maximum no. of epochs to run during training
seed = 42       # seed value to be used throughout training
trn_times = 1   # default is 10 for PascalVOC
run = 1         # run number
eval_interval = max_epochs # eval after x epochs
initialTraining = False
#---------------------------------------------------------------------------#
#----------------- Faster RCNN specific configuration ----------------------#
#---------------------------------------------------------------------------#
optim_lr = 0.001            # optimizer learning rate
optim_weight_decay = 0.0005 # optimizer weight decay
proposals_per_img = 300     # maximum proposals to be generated per image
#--------------------------------------------------------------------------#

#---------------------------------------------------------------------------#
#---------------- Work_dir, Checkpoint & Config file settings --------------#
#---------------------------------------------------------------------------#
root = '../../'
config = '../al_configs/faster_rcnn_r50_fpn_AL_voc0712.py'
base_config = root + '/configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py'
work_dir = '../work_dirs/' + config.split('/')[-1].split('.')[0]
train_script = root + 'tools/train.py'
test_script = root + 'tools/test.py'

if(not(os.path.exists(work_dir))):
    os.makedirs(work_dir)

first_round_checkpoint = work_dir + '/Round_1.pth'
last_epoch_checkpoint = work_dir + '/epoch_' + str(max_epochs) + '.pth'

# set samples_per_gpu & num_gpus such that (samples_per_gpu * num_gpus) is a factor of Active Learning budget
samples_per_gpu = 2     #default is 2
num_gpus = 1            #default is 2
gpu_id =  sys.argv[1]
# if (budget % (samples_per_gpu * num_gpus)) != 0:
#   raise Exception('Budget should be a multiple of samples_per_gpu * no_of_gpus')

#---------------------------------------------------------------------------#
#----------------------- Edit/update Config options ------------------------#
#---------------------------------------------------------------------------#

cfg = Config.fromfile(base_config) # load base config from the base file

cfg_options={}                # edit/update required parms
cfg_options['seed'] = seed
cfg_options['runner.max_epochs'] = max_epochs
cfg_options['data.train.times'] = trn_times
cfg_options['data.samples_per_gpu'] = samples_per_gpu
cfg_options['data.val.ann_file'] = ['trainval_07.txt', 'trainval_12.txt']
cfg_options['data.val.img_prefix'] = copy.deepcopy(cfg.data.train.dataset.img_prefix)
cfg_options['checkpoint_config.interval'] = eval_interval
cfg_options['optimizer.lr'] = optim_lr
cfg_options['optimizer.weight_decay'] = optim_weight_decay
cfg_options['model.train_cfg.rpn_proposal.max_per_img'] = proposals_per_img
cfg_options['model.test_cfg.rpn.max_per_img'] = proposals_per_img
cfg_options['evaluation.interval'] = eval_interval
cfg_options['gpu_ids'] = gpu_id

#---------------------------------------------------------------------------#
#--------------------------- Update Config file ----------------------------#
#---------------------------------------------------------------------------#
cfg.merge_from_dict(cfg_options) # merge with existing config

# write updated configs to a new config file that will be refered for all purposes
file_ptr = open(config, 'w')
file_ptr.write(cfg.pretty_text)
file_ptr.close()

#print(f'Config:\n{cfg.pretty_text}')

#---------------------------------------------------------------------------#
#------------------ Class Imbalance specific setting -----------------------#
#---------------------------------------------------------------------------#
split_cfg = {     
             "per_imbclass_train":10,  # Number of samples per rare class in the train dataset
             "per_imbclass_val":5,     # Number of samples per rare class in the validation dataset
             "per_imbclass_lake":50,   # Number of samples per rare class in the unlabeled dataset
             "per_class_train":100,    # Number of samples per unrare class in the train dataset
             "per_class_val":0,        # Number of samples per unrare class in the validation dataset
             "per_class_lake":500}     # Number of samples per unrare class in the unlabeled dataset

#------------- select imbalanced classes -------------#
imbalanced_classes = [3, 4]     # label of boat & bottle class 


#---------------------------------------------------------------------------#
#------------------------- Build training dataset --------------------------#
#---------------------------------------------------------------------------#
custom_dataset_class = build_dataset_with_indices(RepeatDataset)
trn_dataset = custom_dataset_class(build_dataset(cfg.data.train['dataset'], None), trn_times)

# set total no. of training samples
no_of_trn_samples = len(trn_dataset)
print('No of training samples and budget: ', no_of_trn_samples, budget)

all_classes = set(range(len(trn_dataset.CLASSES)))

#---------------------------------------------------------------------------#
#---- Create Imbalanced Labelled set and Query set from training dataset ---#
#---------------------------------------------------------------------------#

# set the seed to retain order from random selection
np.random.seed(seed)
if(initialTraining):
  # initialize array to contain selected indices from all rounds
  labelled_indices = np.array([])

  # create a random permutation of all training indices
  unlabelled_indices = np.random.permutation(no_of_trn_samples)

  print("#", '-'*15, ' Labelled Dataset Statistics ', '-'*15, "#\n")
  # call custom function to create imbalance & select labelled dataset as per rare & unrare budget
  labelled_indices, unlabelled_indices = create_custom_dataset(trn_dataset, unlabelled_indices, split_cfg['per_imbclass_train'], split_cfg['per_class_train'], imbalanced_classes, all_classes)
  print('\n', len(labelled_indices), " labelled images selected!\n")

  print("#", '-'*15, ' Query Dataset Statistics ', '-'*15, "#\n")
  # call custom function to select query dataset
  query_indices, unlabelled_indices = create_custom_dataset(trn_dataset, unlabelled_indices, split_cfg['per_imbclass_val'], split_cfg['per_class_val'], imbalanced_classes, set(imbalanced_classes))
  print('\n', len(query_indices), " query images selected!")
  print("Query Indices selected: ", query_indices)

  # prepare Validation file from labelled file
  custom_val_file = prepare_val_file(trn_dataset, labelled_indices)

  # set log file
  test_log = open(os.path.join(work_dir,"Round_1_test_mAP.txt"), 'w')

  # save indices in text file for Active Learning
  np.savetxt(os.path.join(work_dir,"labelledIndices.txt"), labelled_indices, fmt='%i')
  np.savetxt(os.path.join(work_dir,"queryIndices.txt"), query_indices, fmt='%i')
  np.savetxt(os.path.join(work_dir,"unlabelledIndices.txt"), unlabelled_indices, fmt='%i')

  # print current selection stats
  labelled_stats = get_class_statistics(trn_dataset, labelled_indices)
  test_log.write("Labelled Dataset Statistics for Round-{}\n".format(str(1)))
  test_log.write('| ' + 'Class'.ljust(10) + 'No. of objects'.ljust(3) + 'No. of images' + '\n')
  test_log.write("-"*40 + '\n')
  for key, val in labelled_stats.items():
    line = '| ' + trn_dataset.CLASSES[key].ljust(15) + str(len(val)).ljust(15) + str(len(set(val)))
    test_log.write(line + '\n')

  #---------------------------------------------------------------------------#
  #----------------------- Call First Round Training -------------------------#
  #---------------------------------------------------------------------------#

  #----- train initial model -----#
  indicesFile = os.path.join(work_dir,"labelledIndices.txt")

  train_command ='python {} {} --indices {}'.format(train_script, config, indicesFile)
  print(train_command)

  for std_out in execute(train_command.split()):
    if std_out[0] != '[':
      print(std_out, end="")

  #----- rename initial model ----#
  copy_command = 'mv {} {}'.format(last_epoch_checkpoint, first_round_checkpoint)
  for std_out in execute(copy_command.split()):
      print(std_out, end="")

  #----- test initial model ------#
  test_command ='python {} {} {} --work-dir {} --eval mAP'.format(test_script, config, first_round_checkpoint, work_dir)
  print(test_command)

  for std_out in execute(test_command.split()):
    if std_out[0] != '[':
      print(std_out, end="")
      test_log.write(std_out)

  test_log.close()
  #------------------------ End of initial training --------------------------#

#---------------------------------------------------------------------------#
#-------------------------------- Run SMI Loop -----------------------------#
#---------------------------------------------------------------------------#

# set SMI parameters
smi_function = sys.argv[2]
if(smi_function == "logdetmi"):
  optimizer = "NaiveGreedy"
else:
  optimizer = "LazyGreedy"
stopIfZeroGain = False
stopIfNegativeGain = False
verbose = False

targeted_cls = imbalanced_classes
strat_dir = os.path.join(work_dir, smi_function, str(run))
    
# create a subdirectory to store log files and data
if(not(os.path.exists(strat_dir))):
    os.makedirs(strat_dir)

# copy labelled, unlabelled indices file from first round backup file. Only these indices are changed in AL rounds
for file in ("labelledIndices.txt", "unlabelledIndices.txt", "queryIndices.txt"):
  src_file = os.path.join(work_dir, file)
  dst_file = os.path.join(strat_dir, file)
  copy_command = 'cp {} {}'.format(src_file, dst_file)
  for std_out in execute(copy_command.split()):
    print(std_out, end="")

# set checkpoint and log file name
last_epoch_checkpoint = strat_dir + '/epoch_' + str(max_epochs) + '.pth'
test_log = open(os.path.join(strat_dir,"SMI_test_mAP.txt"), 'w')

# load from labelled, unlabelled & query indices fies
labelled_indices = np.loadtxt(strat_dir+"/labelledIndices.txt",dtype=int)
unlabelled_indices = np.loadtxt(strat_dir+"/unlabelledIndices.txt",dtype=int)
query_indices = np.loadtxt(strat_dir+"/queryIndices.txt",dtype=int)

#------------ start training for fixed no. of rounds --------------#
for n in range(no_of_rounds-1):
    print("\n","="*20," beginning of round ",n+2," ","="*20,"\n")

    # instantiate the trained model
    if n:
      model = init_detector(config, checkpoint, device='cuda:'+str(gpu_id))
    else:     # for second round, use first round model trained with random indices
      print("second round uses first round model trained with random indices...")
      model = init_detector(config, first_round_checkpoint, device='cuda:'+str(gpu_id))

    # build data loader for unlabelled and query set
    cfg.indices_file = strat_dir + "/unlabelledIndices.txt"
    unlb_loader = build_dataloader(
                  trn_dataset,
                  samples_per_gpu, #cfg.data.samples_per_gpu,
                  cfg.data.workers_per_gpu,
                  # cfg.gpus will be ignored if distributed
                  num_gpus,
                  dist=False,
                  #shuffle=False,
                  seed=cfg.seed,
                  indices_file=cfg.indices_file)

    cfg.indices_file = strat_dir + "/queryIndices.txt"
    query_loader = build_dataloader(
                  trn_dataset,
                  samples_per_gpu, #cfg.data.samples_per_gpu,
                  cfg.data.workers_per_gpu,
                  # cfg.gpus will be ignored if distributed
                  num_gpus,
                  dist=False,
                  #shuffle=False,
                  seed=cfg.seed,
                  indices_file=cfg.indices_file)

    # extract features and compute kernel
    model.eval()
    print("Extracting features for the unlabeled dataset:")
    if(smi_function=="fl1mi" or smi_function=="logdetmi"):
        proposal_budget = 10
        unlabelled_dataset_feat, unlabelled_indices = get_unlabelled_top_k_RoI_features(model, unlb_loader, proposal_budget, feature_type="fc")
        # print("unlabelled_dataset_feat.shape: ", unlabelled_dataset_feat.shape)
        # print("unlabelled indices: ", unlabelled_indices)
    else:
        unlabelled_dataset_feat, unlabelled_indices = get_unlabelled_RoI_features(model, unlb_loader, feature_type="fc")
    print("Extracting features for the query dataset:")
    query_dataset_feat, query_indices = get_query_RoI_features(model, query_loader, imbalanced_classes, feature_type="fc")
    #Free memory
    del model
    del unlb_loader
    del query_loader
    gc.collect()
    
    if(smi_function=="fl1mi" or smi_function=="logdetmi"): # only these smi functions require computing the VxV kernel
        image_image_sim = compute_imageImage_kernel(unlabelled_dataset_feat)
        if(smi_function=="logdetmi"):
          query_query_sim = compute_queryQuery_kernel(query_dataset_feat)
    query_image_sim = compute_queryImage_kernel(query_dataset_feat, unlabelled_dataset_feat) # all functions need the QxV kernel

    # instantiate the submodular functions using the kernels
    if(smi_function =="fl2mi"):
        obj = submodlib.FacilityLocationVariantMutualInformationFunction(n=query_image_sim.shape[1],
                                                              num_queries=query_image_sim.shape[0], 
                                                              query_sijs=query_image_sim.T, 
                                                              queryDiversityEta=1)
    if(smi_function =='gcmi'):
        obj = submodlib.GraphCutMutualInformationFunction(n=query_image_sim.shape[1],
                                                              num_queries=query_image_sim.shape[0], 
                                                              query_sijs=query_image_sim.T)

    if(smi_function =='fl1mi'):
        obj = submodlib.FacilityLocationMutualInformationFunction(n=query_image_sim.shape[1],
                                                                      num_queries=query_image_sim.shape[0], 
                                                                      data_sijs=image_image_sim, 
                                                                      query_sijs=query_image_sim.T, 
                                                                      magnificationEta=1)

    if(smi_function =='logdetmi'):
        obj = submodlib.LogDeterminantMutualInformationFunction(n=query_image_sim.shape[1],
                                                                    num_queries=query_image_sim.shape[0],
                                                                    data_sijs=image_image_sim,  
                                                                    query_sijs=query_image_sim.T,
                                                                    query_query_sijs=query_query_sim,
                                                                    magnificationEta=1,
                                                                    lambdaVal=1)

    greedyList = obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=stopIfZeroGain, 
                              stopIfNegativeGain=stopIfNegativeGain, verbose=verbose)
    # print("greedyList: ", greedyList)
    greedyIndices = [x[0] for x in greedyList]
    greedyIndices = np.array(greedyIndices)
    # print("greedyIndices: ", greedyIndices)
    # print("size of greedy set: ", len(greedyIndices))
    selected_indices = np.array(unlabelled_indices)[greedyIndices]
    # print("selected_indices: ", selected_indices)

    labelled_indices = np.concatenate([labelled_indices, selected_indices])
    unlabelled_indices = np.setdiff1d(unlabelled_indices, selected_indices)
    # print("labeled indices: ", labelled_indices)
    # print("unlabeled indices after setdiff: ", unlabelled_indices)

    # augment rare objects from selected data samples into query set 
    augmented_query_indices, _ = create_custom_dataset(trn_dataset, selected_indices, len(selected_indices) * 1000, 0, imbalanced_classes, set(imbalanced_classes))
    query_indices = np.concatenate([query_indices, augmented_query_indices])
    print("Round ", str(n+2), " dataset statistics:- U: ", len(unlabelled_indices), " L: ", len(labelled_indices), " Q: " , len(query_indices))

    #print(len(unlabelled_indices),len(labelled_indices))
    # save the current list of labelled & unlabelled indices to separate textfiles
    np.savetxt(strat_dir+"/labelledIndices.txt", labelled_indices, fmt='%i')
    np.savetxt(strat_dir+"/unlabelledIndices.txt", unlabelled_indices, fmt='%i')
    np.savetxt(strat_dir+"/queryIndices.txt", query_indices, fmt='%i')

    # print current selection stats
    labelled_stats = get_class_statistics(trn_dataset, labelled_indices)
    test_log.write("Labelled Dataset Statistics for Round-{}\n".format(str(n+2)))
    test_log.write('| ' + 'Class'.ljust(10) + 'No. of objects'.ljust(3) + 'No. of images' + '\n')
    test_log.write("-"*40 + '\n')
    for key, val in labelled_stats.items():
      line = '| ' + trn_dataset.CLASSES[key].ljust(15) + str(len(val)).ljust(15) + str(len(set(val)))
      test_log.write(line + '\n')


    # prepare Validation file from labelled file
    custom_val_file = prepare_val_file(trn_dataset, labelled_indices, strat_dir=strat_dir)

    #----- train current model -----#
    indicesFile = os.path.join(strat_dir,"labelledIndices.txt")

    train_command ='python {} {} --work-dir {} --indices {} --gpu-ids {} --cfg-options'.format(train_script, config, strat_dir, indicesFile, gpu_id)
    train_command = train_command.split()
    train_command.append('data.val.ann_file="{}"'.format(custom_val_file))
    print(' '.join(train_command))

    for std_out in execute(train_command):
      if std_out[0] != '[':
        print(std_out, end="")

    #----- rename initial model ----#
    checkpoint = strat_dir + '/Round_' + str(n+2) + '.pth'  # set checkpoint file path
    copy_command = 'mv {} {}'.format(last_epoch_checkpoint, checkpoint)
    for std_out in execute(copy_command.split()):
      print(std_out, end="")

    #----- test initial model ------#
    test_command ='python {} {} {} --work-dir {} --eval mAP'.format(test_script, config, checkpoint, strat_dir)
    print(test_command)

    for std_out in execute(test_command.split()):
      if std_out[0] != '[':
        print(std_out, end="")
        test_log.write(std_out)
    
    #--------------------------- End of current round -----------------------------#
test_log.close()