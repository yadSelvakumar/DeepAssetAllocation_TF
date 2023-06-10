from argparse import ArgumentParser, Namespace

parser = ArgumentParser('DeepAssetAllocation')
# parser.add_argument("--nt", type=int, default=2, help="depth of networks")
# parser.add_argument("--width", type=int, default=16, help="width of networks")
# parser.add_argument("--batch_size_alpha", type=int, default=1024, help="batch size for training alpha")
# parser.add_argument("--max_steps_alpha", type=int, default=2048, help="maximum number of optimization steps for alpha")
# parser.add_argument("--lr_alpha", type=float, default=1e-3, help="learning rate for alpha")
# parser.add_argument("--batch_size_NN", type=int, default=16, help="batch size for neural network training")
# parser.add_argument("--max_epochs_NN", type=int, default=1024, help="maximum number of epochs for neural network training")
# parser.add_argument("--lr_NN", type=float, default=1e-3, help="learning rate for neural network training")
# parser.add_argument("--num_samples_JV", type=int, default=1024, help="number of samples to validate JV solution")
parser.add_argument("--num_samples", type=int, default=4096, help="number of training trajectories")
parser.add_argument('--results_dir', type=str, default='results', help='directory for results')
parser.add_argument('--figures_dir', type=str, default='figures', help='directory for figures')
parser.add_argument('--logs_dir', type=str, default='logs', help='directory for logs')
parser.add_argument('--settings_file', type=str, default='settings/ResultsForYad.mat', help='matlab settings file')
# parser.add_argument('--constraint', type=str, choices=['no_short_selling', 'budget', 'box'], default='budget', help='constraints for alpha')

args: Namespace  = parser.parse_args()

if __name__ == '__main__':
    from src.training import train_model
    train_model(args)
