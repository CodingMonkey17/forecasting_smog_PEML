import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader
from modelling.loss import *
import time  # Import time module to measure training time and inference time


class BasicMLP(nn.Module):
    def __init__(self, N_INPUT_UNITS = 72, N_HIDDEN_LAYERS = 2, N_HIDDEN_UNITS = 100, N_OUTPUT_UNITS = 24, loss_function="MSE"):
        super(BasicMLP, self).__init__()
        self.loss_function = loss_function  # Store the loss function type

        layers = [nn.Linear(N_INPUT_UNITS, N_HIDDEN_UNITS), nn.ReLU()]

        for _ in range(N_HIDDEN_LAYERS):
            layers.append(nn.Linear(N_HIDDEN_UNITS, N_HIDDEN_UNITS))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(N_HIDDEN_UNITS, N_OUTPUT_UNITS))
        self.layers = nn.Sequential(*layers)

    def forward(self, u):
        batch_size, seq_length, n_features = u.size()
        outputs = torch.empty((batch_size, seq_length - 1, self.layers[-1].out_features), device=u.device)

        for t in range(1, seq_length):  # Start from t=1 to use t-1 as input
            u_t = u[:, t-1, :]  # Use previous timestep input
            y_t = self.layers(u_t)
            outputs[:, t-1, :] = y_t  # Save at t-1 because first output corresponds to t=1

        return outputs[:, -24:, :]  # Predict last 24 hours

    def train_model(self, train_loader, val_loader, all_y_phy = None, epochs=50, lr=1e-3, weight_decay=1e-6, lambda_phy = 1e-5, device="cpu", trial=None):
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_loss = float("inf")
        best_model_state = None
        # Start timing training

        start_train_time = time.time()

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            # load y_phy file 
            print(f"Epoch {epoch+1}/{epochs}")

            #loop with batch idx too
            for batch_idx, (u, y) in enumerate(train_loader):
                u, y = u.to(device), y.to(device)
                optimizer.zero_grad()

                output = self.forward(u)
                loss= compute_loss(output, y, u, self.loss_function, lambda_phy = lambda_phy, all_y_phy = all_y_phy, batch_idx = batch_idx)  # Compute loss based on selected function

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)


            # Validation step
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for u, y in val_loader:
                    u, y = u.to(device), y.to(device)
                    output = self.forward(u)
                    mse_loss_fn = nn.MSELoss()
                    mse_loss = mse_loss_fn(output, y)
                    #val loss is rmse
                    val_loss += torch.sqrt(mse_loss).item()

            val_loss /= len(val_loader)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.state_dict()

            if trial is not None:  # If using Optuna for hyperparameter tuning
                trial.report(val_loss, step=epoch)  # Report the validation loss to Optuna
                if trial.should_prune():
                    raise optuna.TrialPruned()

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss (simple RMSE, no physics involved): {val_loss:.6f}")


        total_train_time = time.time() - start_train_time  # Total training time
        if best_model_state:
            self.load_state_dict(best_model_state)
        return best_val_loss, total_train_time


    def test_model(self, test_loader, min_value=None, max_value=None, device="cpu"):
        self.to(device)
        self.eval()
        mse_loss_fn =nn.MSELoss(reduction="mean")
        rmse_loss = 0.0
        smape_loss = 0.0
        total_elements = 0
        epsilon = 1e-6  # To prevent division by zero
        total_mse_loss = 0.0 
        start_test_time = time.time()  # Start timing inference

        with torch.no_grad():
            for u, y in test_loader:
                u, y = u.to(device), y.to(device)
                output = self.forward(u)

                # Compute SMAPE BEFORE denormalization
                abs_diff = torch.abs(y - output)
                sum_abs = torch.abs(y) + torch.abs(output) + epsilon  # Avoid division by zero
                smape_batch = torch.sum(2 * abs_diff / sum_abs).item()

                smape_loss += smape_batch
                total_elements += y.numel()  # Count total number of elements

                # Denormalize for RMSE and MSE calculation
                if min_value is not None and max_value is not None:
                    output = output * (max_value - min_value) + min_value
                    y = y * (max_value - min_value) + min_value
                output = output.to(torch.float64)
                y = y.to(torch.float64)

                mse_loss = mse_loss_fn(output, y)
                total_mse_loss += mse_loss.item()  # Sum up MSE
        
        # Final RMSE calculation: take the square root of the mean MSE
        mean_mse = total_mse_loss / len(test_loader)  # Mean MSE over batches
        rmse_loss = mean_mse ** 0.5

        smape_loss = (smape_loss / total_elements) * 100  # Average over all elements and convert to %

        total_test_time = time.time() - start_test_time  # Total inference time
        
        print(f"Test MSE Loss: {mse_loss.item():.6f}")
        print(f"Test RMSE Loss: {rmse_loss:.6f}")
        print(f"Test SMAPE Loss: {smape_loss:.6f}%")
        print(f"Total Inference Time: {total_test_time:.2f} seconds")

        return mse_loss.item(), rmse_loss, smape_loss, total_test_time
