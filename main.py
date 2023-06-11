from argparse import ArgumentParser, Namespace

parser = ArgumentParser('DeepAssetAllocation')
parser.add_argument("--num_samples", type=int, default=4096, help="number of training trajectories")
parser.add_argument("--batch_size", type=int, default=1024, help="size of the batches")

parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate for training")
parser.add_argument("--decay_steps", type=float, default=.5, help="decay of learning rate steps for training")
parser.add_argument("--decay_rate", type=int, default=400, help="decay of learning rate for training")

parser.add_argument('--results_dir', type=str, default='results', help='directory for results')
parser.add_argument('--figures_dir', type=str, default='figures', help='directory for figures')
parser.add_argument('--logs_dir', type=str, default='logs', help='directory for logs')

parser.add_argument('--settings_file', type=str, default='settings/ResultsForYad.mat', help='matlab settings file')
# parser.add_argument('--constraint', type=str, choices=['no_short_selling', 'budget', 'box'], default='budget', help='constraints for alpha')

args: Namespace  = parser.parse_args()

if __name__ == '__main__':
    from src.training import train_model
    train_model(args)
