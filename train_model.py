from argparse import ArgumentParser

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
# parser.add_argument("--num_train", type=int, default=4096, help="number of training trajectories")
parser.add_argument('--out_dir', type=str, default='test', help='directory for result')
# parser.add_argument('--out_file', type=str, default='test', help='base file name for result')
# parser.add_argument('--figures_out_dir', type=str, default='figures', help='directory for figures')
# parser.add_argument('--constraint', type=str, choices=['no_short_selling', 'budget', 'box'], default='budget', help='constraints for alpha')

args = parser.parse_args()

# TODO: main file run with arguments
# TODO: dynamic dir system

if __name__ == '__main__':
    from training import main
    main(args)
