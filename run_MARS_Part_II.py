from argparse import ArgumentParser, Namespace
from src.utils import check_est_flag
from src.main_scripts import main_full_model, main_allocation_daily

settings_file = 'settings/qmas_bt_20230627_with_alloc.mat'
nn_saved_file = 'results_save'
estimation_flag = check_est_flag(settings_file)

if estimation_flag==1:
    main_full_model(settings_file)
    main_allocation_daily(settings_file,nn_saved_file)
elif estimation_flag==0:
    #TO ABAKE: IF HERE, 'nn_saved_file' WILL NEED TO REFERENCE THE LAST SAVED RESULTS FOLDER
    main_allocation_daily(settings_file,nn_saved_file)
