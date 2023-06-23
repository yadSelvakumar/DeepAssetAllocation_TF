from argparse import ArgumentParser, Namespace

parser = ArgumentParser('DeepAssetAllocationTraining')
parser.add_argument("--num_samples", type=int, default=4096, help="number of training trajectories")
parser.add_argument("--batch_size", type=int, default=1024, help="size of the batches")#1024

parser.add_argument("--learning_rate_alpha", type=float, default=1e-3, help="learning rate for alpha training")
parser.add_argument("--decay_steps_alpha", type=int, default=625, help="decay of learning rate steps for alpha training")  # 400
parser.add_argument("--decay_rate_alpha", type=float, default=.5, help="decay of learning rate for alpha training")

parser.add_argument("--iter_per_epoch", type=int, default=50, help="alpha iterations per epoch")
parser.add_argument("--num_epochs_alpha", type=int, default=50, help="alpha number of epochs") #50
parser.add_argument("--num_epochs", type=int, default=25_000, help="number of epochs training")#100_000
parser.add_argument('--alpha_constraint', type=str, choices=['retail-relu', 'sum-to-1'], default='sum-to-1', help='constraints for alpha')

parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate for training")
parser.add_argument("--decay_steps", type=int, default=7_000, help="decay of learning rate steps for training")#
parser.add_argument("--decay_rate", type=float, default=.5, help="decay of learning rate for training")

parser.add_argument('--model_output_size', type=int, default=1, help='output size of the model')
parser.add_argument('--num_hidden_layers', type=int, default=2, help='number of hidden layers')
parser.add_argument('--num_neurons', type=int, default=32, help='number of neurons per hidden layer')
parser.add_argument('--activation_function', type=str, choices=['tanh', 'relu', 'elu'], default='elu', help='activation function for hidden layers')
parser.add_argument('--activation_function_output', type=str, choices=['tanh', 'relu', 'linear'], default='linear', help='activation function for output layer')
parser.add_argument('--initial_guess', type=float, default=1, help='initial guess for model')

parser.add_argument('--results_dir', type=str, default='results', help='directory for results')
parser.add_argument('--figures_dir', type=str, default='figures', help='directory for figures')
parser.add_argument('--results_dir_save', type=str, default='results_save', help='directory for results to save')
parser.add_argument('--figures_dir_save', type=str, default='figures_save', help='directory for figures to save')
parser.add_argument('--logs_dir', type=str, default='logs', help='directory for logs')

parser.add_argument('--settings_file', type=str, default='settings/model_settings_la_caixa_new.mat', help='matlab settings file')

args: Namespace = parser.parse_args()

if __name__ == '__main__':
    from src.training import train_model
    train_model(args)