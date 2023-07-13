import os
from argparse import ArgumentParser, Namespace

def main_full_model(setting_file, HOME_DIR):

    parser = ArgumentParser('DeepAssetAllocationTraining')
    parser.add_argument("--num_samples", type=int, default=4096, help="number of training trajectories")
    parser.add_argument("--batch_size", type=int, default=1024, help="size of the batches")

    parser.add_argument("--learning_rate_alpha", type=float, default=1e-3, help="learning rate for alpha training")
    parser.add_argument("--decay_steps_alpha", type=int, default=625, help="decay of learning rate steps for alpha training")  # 625
    parser.add_argument("--decay_rate_alpha", type=float, default=.5, help="decay of learning rate for alpha training")

    parser.add_argument("--iter_per_epoch", type=int, default=50, help="alpha iterations per epoch")
    parser.add_argument("--num_epochs_alpha", type=int, default=50, help="alpha number of epochs")
    parser.add_argument("--num_epochs", type=int, default=100_000, help="number of epochs training")
    parser.add_argument('--alpha_constraint', type=str, choices=['retail-relu', 'sum-to-1'], default='sum-to-1', help='constraints for alpha')

    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate for training")
    parser.add_argument("--decay_steps", type=int, default=25_000, help="decay of learning rate steps for training")
    parser.add_argument("--decay_rate", type=float, default=.5, help="decay of learning rate for training")

    parser.add_argument('--model_output_size', type=int, default=1, help='output size of the model')
    parser.add_argument('--num_hidden_layers', type=int, default=2, help='number of hidden layers')
    parser.add_argument('--num_neurons', type=int, default=32, help='number of neurons per hidden layer')
    parser.add_argument('--activation_function', type=str, choices=['tanh', 'relu', 'elu'], default='elu', help='activation function for hidden layers')
    parser.add_argument('--activation_function_output', type=str, choices=['tanh', 'relu', 'linear'], default='linear', help='activation function for output layer')
    parser.add_argument('--initial_guess', type=float, default=1, help='initial guess for model')

    parser.add_argument('--results_dir', type=str, default=os.path.join(HOME_DIR, 'results'), help='directory for results')
    parser.add_argument('--figures_dir', type=str, default=os.path.join(HOME_DIR, 'figures'), help='directory for figures')
    parser.add_argument('--results_dir_save', type=str, default=os.path.join(HOME_DIR, 'results_save'), help='directory for results to save')
    parser.add_argument('--figures_dir_save', type=str, default=os.path.join(HOME_DIR, 'figures_save'), help='directory for figures to save')
    parser.add_argument('--logs_dir', type=str, default=os.path.join(HOME_DIR, 'logs'), help='directory for logs')
    parser.add_argument('--plot_toggle', type=int, default=1, help='indicator to print figures')

    parser.add_argument('--settings_file', type=str, default=setting_file, help='matlab settings file')

    args: Namespace = parser.parse_args()
    
    # ------------------------------- Execute code ------------------------------- #
    from src.training import train_model
    train_model(args)


def main_allocation_daily(settings_file, nn_saved_file, HOME_DIR):

    parser = ArgumentParser('DeepAssetAllocationTraining')
    parser.add_argument("--batch_size", type=int, default=1024, help="size of the batches")

    parser.add_argument("--learning_rate_alpha", type=float, default=1e-3, help="learning rate for alpha training")
    parser.add_argument("--decay_steps_alpha", type=int, default=3000, help="first period decay of learning rate steps for alpha training")  # 400
    parser.add_argument("--decay_rate_alpha", type=float, default=.5, help="decay of learning rate for alpha training")

    parser.add_argument("--iter_per_epoch", type=int, default=50, help="alpha iterations per epoch")
    parser.add_argument("--num_epochs_alpha", type=int, default=256, help="alpha number of epochs")
    parser.add_argument("--num_epochs", type=int, default=100_000, help="number of epochs training")
    parser.add_argument('--alpha_constraint', type=str, choices=['retail-relu', 'sum-to-1'], default='retail-relu', help='constraints for alpha')

    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate for training")
    parser.add_argument("--decay_steps", type=int, default=25_000, help="first period decay of learning rate steps for training")
    parser.add_argument("--decay_rate", type=float, default=.5, help="decay of learning rate for training")

    parser.add_argument('--model_output_size', type=int, default=1, help='output size of the model')
    parser.add_argument('--num_hidden_layers', type=int, default=2, help='number of hidden layers')
    parser.add_argument('--num_neurons', type=int, default=32, help='number of neurons per hidden layer')
    parser.add_argument('--activation_function', type=str, choices=['tanh', 'relu', 'elu'], default='elu', help='activation function for hidden layers')
    parser.add_argument('--activation_function_output', type=str, choices=['tanh', 'relu', 'linear'], default='linear', help='activation function for output layer')
    parser.add_argument('--initial_guess', type=float, default=1, help='initial guess for model')

    parser.add_argument('--results_dir', type=str, default=os.path.join(HOME_DIR, 'results'), help='directory for results')
    parser.add_argument('--figures_dir', type=str, default=os.path.join(HOME_DIR, 'figures'), help='directory for figures')
    parser.add_argument('--results_dir_save', type=str, default=nn_saved_file, help='directory for results to save')
    parser.add_argument('--figures_dir_save', type=str, default=os.path.join(HOME_DIR, 'figures_save'), help='directory for figures to save')
    parser.add_argument('--logs_dir', type=str, default=os.path.join(HOME_DIR, 'logs'), help='directory for logs')

    parser.add_argument('--settings_file', type=str, default=settings_file, help='matlab settings file')

    args: Namespace = parser.parse_args()

    # ------------------------------- Execute code ------------------------------- #
    from src.calc_weights_realtime import calc_fixed_horizon_allocations, calc_term_fund_allocations, save_results
    import pandas as pd
    date_today = pd.to_datetime('today').strftime("%Y-%m-%d")
    alphas_tactical_fixed_horizon, alphas_strategic_fixed_horizon, alphas_tactical_JV_fixed_horizon, alphas_strategic_JV_fixed_horizon, date_fixed_horizon, invest_horizon_fixed_horizon, data_t, data_tplus1, unconditional_mean = calc_fixed_horizon_allocations(args, invest_horizon=48)
    alphas_tactical_target_date, alphas_strategic_target_date, alphas_tactical_JV_target_date, alphas_strategic_JV_target_date, date_target_date, invest_horizon_target_date =  calc_term_fund_allocations(args, term_date='12-31-2025')
    save_results(args.results_dir_save, f'check_real_time_allocations_{date_today}',alphas_tactical_fixed_horizon, alphas_strategic_fixed_horizon, alphas_tactical_JV_fixed_horizon, alphas_strategic_JV_fixed_horizon, date_fixed_horizon, invest_horizon_fixed_horizon, alphas_tactical_target_date, alphas_strategic_target_date, alphas_tactical_JV_target_date, alphas_strategic_JV_target_date, date_target_date, invest_horizon_target_date, data_t, data_tplus1, unconditional_mean)


def main_allocation_full_sample(settings_file, nn_saved_file, HOME_DIR):
    parser = ArgumentParser('DeepAssetAllocationTraining')
    parser.add_argument("--batch_size", type=int, default=1024, help="size of the batches")

    parser.add_argument("--learning_rate_alpha", type=float, default=1e-3, help="learning rate for alpha training")
    parser.add_argument("--decay_steps_alpha", type=int, default=3000, help="first period decay of learning rate steps for alpha training")  # 400
    parser.add_argument("--decay_rate_alpha", type=float, default=.5, help="decay of learning rate for alpha training")

    parser.add_argument("--iter_per_epoch", type=int, default=50, help="alpha iterations per epoch")
    parser.add_argument("--num_epochs_alpha", type=int, default=256, help="alpha number of epochs")
    parser.add_argument("--num_epochs", type=int, default=100_000, help="number of epochs training")
    parser.add_argument('--alpha_constraint', type=str, choices=['retail-relu', 'sum-to-1'], default='retail-relu', help='constraints for alpha')

    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate for training")
    parser.add_argument("--decay_steps", type=int, default=25_000, help="first period decay of learning rate steps for training")
    parser.add_argument("--decay_rate", type=float, default=.5, help="decay of learning rate for training")

    parser.add_argument('--model_output_size', type=int, default=1, help='output size of the model')
    parser.add_argument('--num_hidden_layers', type=int, default=2, help='number of hidden layers')
    parser.add_argument('--num_neurons', type=int, default=32, help='number of neurons per hidden layer')
    parser.add_argument('--activation_function', type=str, choices=['tanh', 'relu', 'elu'], default='elu', help='activation function for hidden layers')
    parser.add_argument('--activation_function_output', type=str, choices=['tanh', 'relu', 'linear'], default='linear', help='activation function for output layer')
    parser.add_argument('--initial_guess', type=float, default=1, help='initial guess for model')

    parser.add_argument('--results_dir', type=str, default=os.path.join(HOME_DIR, 'results'), help='directory for results')
    parser.add_argument('--figures_dir', type=str, default=os.path.join(HOME_DIR, 'figures'), help='directory for figures')
    parser.add_argument('--results_dir_save', type=str, default=nn_saved_file, help='directory for results to save')
    parser.add_argument('--figures_dir_save', type=str, default=os.path.join(HOME_DIR, 'figures_save'), help='directory for figures to save')
    parser.add_argument('--logs_dir', type=str, default=os.path.join(HOME_DIR, 'logs'), help='directory for logs')
    parser.add_argument('--plot_toggle', type=int, default=1, help='indicator to print figures')

    parser.add_argument('--settings_file', type=str, default=settings_file, help='matlab settings file')

    args: Namespace = parser.parse_args()

    # ------------------------------- Execute code ------------------------------- #
    from src.calc_weights_full_sample import calc_fixed_horizon_allocations, calc_term_fund_allocations, save_results
    import pandas as pd
    date_today = pd.to_datetime('today').strftime("%Y-%m-%d")
    alphas_tactical_fixed_horizon, alphas_strategic_fixed_horizon, alphas_tactical_JV_fixed_horizon, alphas_strategic_JV_fixed_horizon, date_fixed_horizon, invest_horizon_fixed_horizon, unconditional_mean = calc_fixed_horizon_allocations(args, invest_horizon=48)
    alphas_tactical_target_date, alphas_strategic_target_date, alphas_tactical_JV_target_date, alphas_strategic_JV_target_date, date_target_date, invest_horizon_target_date =  calc_term_fund_allocations(args, term_date='12-31-2024')
    save_results(args.results_dir_save, f'real_time_allocations_{date_today}',alphas_tactical_fixed_horizon, alphas_strategic_fixed_horizon, alphas_tactical_JV_fixed_horizon, alphas_strategic_JV_fixed_horizon, date_fixed_horizon, invest_horizon_fixed_horizon, alphas_tactical_target_date, alphas_strategic_target_date, alphas_tactical_JV_target_date, alphas_strategic_JV_target_date, date_target_date, invest_horizon_target_date, unconditional_mean)
