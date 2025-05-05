import torch
import torch.nn as nn
import torch.optim as optim
import time
import optuna
from modelling.loss import *

class GRU(nn.Module):
    def __init__(self,
                 N_HOURS_U=72,
                 N_HOURS_Y=24,
                 N_INPUT_UNITS=10,
                 N_HIDDEN_LAYERS=2,
                 N_HIDDEN_UNITS=128,
                 N_OUTPUT_UNITS=4,
                 loss_function="MSE"):
        super(GRU, self).__init__()
        self.loss_function = loss_function

        self.d_n_hours_u = N_HOURS_U
        self.d_n_hours_y = N_HOURS_Y
        self.d_input_units = N_INPUT_UNITS
        self.d_hidden_layers = N_HIDDEN_LAYERS
        self.d_hidden_units = N_HIDDEN_UNITS
        self.d_output_units = N_OUTPUT_UNITS

        self.gru = nn.GRU(
            input_size=self.d_input_units,
            hidden_size=self.d_hidden_units,
            num_layers=self.d_hidden_layers,
            batch_first=True
        )
        self.dense = nn.Linear(self.d_hidden_units, self.d_output_units)
    
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GRU:
        - Pass input tensor through GRU neurons
        - For each prediction time step t, use hidden state from t-1
        - Apply dense layer to generate predictions
        
        This avoids data leakage by ensuring predictions for time t
        only use information available up to time t-1.
        """
        out, _ = self.gru(u)  # shape: [batch_size, sequence_length, hidden_size]
        
        # Get the timesteps used for predictions (all except the last N_HOURS_Y)
        # We use these to predict the next N_HOURS_Y steps
        pred_indices = range(self.d_n_hours_u - self.d_n_hours_y, self.d_n_hours_u)
        
        # For each prediction step, use the previous timestep's hidden state
        predictions = []
        for idx in pred_indices:
            # Use hidden state at position idx to predict the next timestep
            pred = self.dense(out[:, idx-1, :])
            predictions.append(pred)
        
        # Stack predictions along a new dimension and reshape to 
        # [batch_size, n_hours_y, n_output_units]
        return torch.stack(predictions, dim=1)

    def train_model(self, train_loader, val_loader, all_y_phy = None, epochs=50, lr=1e-3, weight_decay=1e-6, lambda_phy = 1e-5, device="cpu", trial=None, 
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
        early_stop_patience = 15
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
                loss= compute_loss(output, y, u, self.loss_function, lambda_phy = lambda_phy, all_y_phy = all_y_phy, batch_idx = batch_idx, train_loader= train_loader, 
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
