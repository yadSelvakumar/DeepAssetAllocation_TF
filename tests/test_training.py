from old_training import train_model as old_train_model
from src.training import train_model
import numpy as np


def test_training(args):
    total_time, losses = old_train_model(args, 2) 
    U_total_time, U_losses = train_model(args, 2)

    assert U_losses[-1] < losses[-1] or abs(losses[-1] - U_losses[-1]) < 1e-3
    assert U_losses[-1] < losses[-1] or abs(losses[-2] - U_losses[-2]) < 1e-3
    u_mean_loss, mean_loss = np.mean(U_losses), np.mean(losses)
    assert u_mean_loss < mean_loss or abs(u_mean_loss - mean_loss) < 1e-3
    assert U_total_time < total_time or abs(total_time - U_total_time) < 2
