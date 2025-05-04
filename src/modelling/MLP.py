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

    def train_model(self, train_loader, val_loader, all_y_phy = None, epochs=50, lr=1e-3, weight_decay=1e-6, lambda_phy = 1e-5, lambda_ic =0, k= 2, D= 0.1, device="cpu", trial=None, 
                    station_names = ['tuindorp', 'breukelen'], main_station = 'breukelen', idx_dict = None):
        
        print(f"Using stations {station_names} for input and {main_station} as the main predicting station")
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


        # Add scheduler according to Table D.2
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.1,  # Reduce LR by a factor of 0.1
            patience=5,  # Wait for 5 epochs of no improvement
            verbose=True
        )
        
        # Add early stopping with patience=6 as per Table D.2
        early_stop_patience = 6
        no_improve_count = 0

        best_val_loss = float("inf")
        best_model_state = None
        train_losses = []
        val_losses = []
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
                loss= compute_loss(output, y, u, self.loss_function, lambda_phy = lambda_phy, lambda_ic= lambda_ic, k=k, D=D, all_y_phy = all_y_phy, batch_idx = batch_idx, train_loader= train_loader, 
                                   idx_dict = idx_dict ,station_names= station_names, main_station = main_station)  # Compute loss based on selected function

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)


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
                    val_loss += mse_loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            # Update scheduler based on validation loss
            scheduler.step(val_loss)
            
            # Save best model and check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.state_dict()
                no_improve_count = 0  # Reset counter
            else:
                no_improve_count += 1  # Increment counter
                
            # Apply early stopping if needed
            if no_improve_count >= early_stop_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
                
            # Optuna handling
            if trial is not None:
                trial.report(val_loss, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            # current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")


        total_train_time = time.time() - start_train_time  # Total training time
        if best_model_state:
            self.load_state_dict(best_model_state)
        return best_val_loss, total_train_time, train_losses, val_losses


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
                    min_value_tensor = torch.tensor(min_value, device=device, dtype=output.dtype)
                    max_value_tensor = torch.tensor(max_value, device=device, dtype=output.dtype)
                    output = output * (max_value_tensor - min_value_tensor) + min_value_tensor
                    y = y * (max_value_tensor - min_value_tensor) + min_value_tensor
                output = output.to(torch.float64)
                y = y.to(torch.float64)

                mse_loss = mse_loss_fn(output, y)
                total_mse_loss += mse_loss.item()  # Sum up MSE
        
        # Final RMSE calculation: take the square root of the mean MSE
        mean_mse = total_mse_loss / len(test_loader)  # Mean MSE over batches
        rmse_loss = mean_mse ** 0.5

        
        smape_loss = (smape_loss / total_elements) * 100  # Average over all elements and convert to %

        total_test_time = time.time() - start_test_time  # Total inference time
        
        print(f"Test MSE Loss: {mean_mse:.6f}")
        print(f"Test RMSE Loss: {rmse_loss:.6f}")
        print(f"Test SMAPE Loss: {smape_loss:.6f}%")
        print(f"Total Inference Time: {total_test_time:.2f} seconds")

        return mean_mse, rmse_loss, smape_loss, total_test_time
