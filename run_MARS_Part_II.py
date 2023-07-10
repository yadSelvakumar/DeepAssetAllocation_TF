import os

os.environ['CUDA_CACHE_DISABLE'] = '0'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '1'
os.environ['TF_ADJUST_HUE_FUSED'] = '1'
os.environ['TF_ADJUST_SATURATION_FUSED'] ='1'
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ['TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32'] = '1'
os.environ['TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32'] = '1'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE'] = '1'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION_LOSS_SCALING'] = '1'
os.environ['TF_ENABLE_XLA'] = '1'

HOME_DIR = os.path.dirname(os.path.abspath(__file__))

from src.utils import check_est_flag
from src.main_scripts import main_full_model, main_allocation_daily

settings_file = os.path.join(HOME_DIR, 'settings/qmas_bt_20230627_with_alloc.mat')
nn_saved_file = os.path.join(HOME_DIR, 'results_save')
estimation_flag = check_est_flag(settings_file)

if estimation_flag==1:
    main_full_model(settings_file, HOME_DIR)
    main_allocation_daily(settings_file, nn_saved_file, HOME_DIR)
elif estimation_flag==0:
    #TO ABAKE: IF HERE, 'nn_saved_file' WILL NEED TO REFERENCE THE LAST SAVED RESULTS FOLDER
    main_allocation_daily(settings_file,nn_saved_file, HOME_DIR)
