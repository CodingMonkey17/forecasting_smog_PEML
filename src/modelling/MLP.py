import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader
from modelling.loss import *


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
        outputs = torch.empty((batch_size, seq_length, self.layers[-1].out_features), device=u.device)

        for t in range(seq_length):
            u_t = u[:, t, :]
            y_t = self.layers(u_t)
            outputs[:, t, :] = y_t

        return outputs[:, -24:, :]  # Predict last 24 hours

    def train_model(self, train_loader, val_loader, epochs=50, lr=1e-3, weight_decay=1e-6, device="cpu"):
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_loss = float("inf")
        best_model_state = None

        train_losses = []
        val_losses = []
        
        # Initialize lists to store y_phy and y_pred for later plotting
        all_y_phy_train = []
        all_y_pred_train = []
        all_y_phy_val = []
        all_y_pred_val = []

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            print(f"Epoch {epoch+1}/{epochs}")
            for u, y in train_loader:
                u, y = u.to(device), y.to(device)
                optimizer.zero_grad()

                output = self.forward(u)
                
                loss, y_phy, y_pred = compute_loss(output, y, u, self.loss_function, lambda_phy = 1e-5)  # Compute loss based on selected function
                
                # Append y_phy and y_pred to lists for training data
                all_y_phy_train.append(y_phy.cpu().detach().numpy())  # .cpu() to move to CPU, .detach() to avoid tracking gradients
                all_y_pred_train.append(y_pred.cpu().detach().numpy())

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
                    loss, y_phy, y_pred = compute_loss(output, y, u, self.loss_function)

                    # Append y_phy and y_pred to lists for validation data
                    all_y_phy_val.append(y_phy.cpu().detach().numpy())
                    all_y_pred_val.append(y_pred.cpu().detach().numpy())

                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.state_dict()

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")

        if best_model_state:
            self.load_state_dict(best_model_state)

        # Flatten the list of values for easier plotting
        all_y_phy_train = np.concatenate(all_y_phy_train)
        all_y_pred_train = np.concatenate(all_y_pred_train)
        all_y_phy_val = np.concatenate(all_y_phy_val)
        all_y_pred_val = np.concatenate(all_y_pred_val)

        return best_val_loss, train_losses, val_losses, all_y_phy_train, all_y_pred_train, all_y_phy_val, all_y_pred_val


    def test_model(self, test_loader, min_value=None, max_value=None, device="cpu"):
        self.to(device)
        self.eval()
        mse_loss_fn = nn.MSELoss()
        rmse_loss = 0.0
        smape_loss = 0.0
        total_elements = 0
        epsilon = 1e-6  # To prevent division by zero

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

                mse_loss = mse_loss_fn(output, y)
                rmse_loss += torch.sqrt(mse_loss).item()

        # Final loss calculations
        rmse_loss /= len(test_loader)  # Average over batches
        smape_loss = (smape_loss / total_elements) * 100  # Average over all elements and convert to %

        print(f"Test MSE Loss: {mse_loss.item():.6f}")
        print(f"Test RMSE Loss: {rmse_loss:.6f}")
        print(f"Test SMAPE Loss: {smape_loss:.6f}%")

        return mse_loss.item(), rmse_loss, smape_loss
