# Difference to v03: all model reactions

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from datetime import date

datafile = "./data/2025-06-23_full_training_data_149261_samples.csv"

class MetabolicNN(nn.Module):
    """Neural network to predict metabolic fluxes"""
    def __init__(self, input_size=20, hidden_size=256, output_size=95):
        super(MetabolicNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.model(x)

def load_and_preprocess_data(filename):
    """Load and preprocess the training data"""
    input_cols = [
        'EX_glc__D_e', 'EX_fru_e', 'EX_lac__D_e', 'EX_pyr_e', 'EX_ac_e',
        'EX_akg_e', 'EX_succ_e', 'EX_fum_e', 'EX_mal__L_e', 'EX_etoh_e',
        'EX_acald_e', 'EX_for_e', 'EX_gln__L_e', 'EX_glu__L_e',
        'EX_co2_e', 'EX_h_e', 'EX_h2o_e', 'EX_nh4_e', 'EX_o2_e', 'EX_pi_e'
    ]

    output_cols = [
        'ACALD_flux',
        'ACALDt_flux',
        'ACKr_flux',
        'ACONTa_flux',
        'ACONTb_flux',
        'ACt2r_flux',
        'ADK1_flux',
        'AKGDH_flux',
        'AKGt2r_flux',
        'ALCD2x_flux',
        'ATPM_flux',
        'ATPS4r_flux',
        'Biomass_Ecoli_core_flux',
        'CO2t_flux',
        'CS_flux',
        'CYTBD_flux',
        'D_LACt2_flux',
        'ENO_flux',
        'ETOHt2r_flux',
        'EX_ac_e_flux',
        'EX_acald_e_flux',
        'EX_akg_e_flux',
        'EX_co2_e_flux',
        'EX_etoh_e_flux',
        'EX_for_e_flux',
        'EX_fru_e_flux',
        'EX_fum_e_flux',
        'EX_glc__D_e_flux',
        'EX_gln__L_e_flux',
        'EX_glu__L_e_flux',
        'EX_h_e_flux',
        'EX_h2o_e_flux',
        'EX_lac__D_e_flux',
        'EX_mal__L_e_flux',
        'EX_nh4_e_flux',
        'EX_o2_e_flux',
        'EX_pi_e_flux',
        'EX_pyr_e_flux',
        'EX_succ_e_flux',
        'FBA_flux',
        'FBP_flux',
        'FORt2_flux',
        'FORti_flux',
        'FRD7_flux',
        'FRUpts2_flux',
        'FUM_flux',
        'FUMt2_2_flux',
        'G6PDH2r_flux',
        'GAPD_flux',
        'GLCpts_flux',
        'GLNS_flux',
        'GLNabc_flux',
        'GLUDy_flux',
        'GLUN_flux',
        'GLUSy_flux',
        'GLUt2r_flux',
        'GND_flux',
        'H2Ot_flux',
        'ICDHyr_flux',
        'ICL_flux',
        'LDH_D_flux',
        'MALS_flux',
        'MALt2_2_flux',
        'MDH_flux',
        'ME1_flux',
        'ME2_flux',
        'NADH16_flux',
        'NADTRHD_flux',
        'NH4t_flux',
        'O2t_flux',
        'PDH_flux',
        'PFK_flux',
        'PFL_flux',
        'PGI_flux',
        'PGK_flux',
        'PGL_flux',
        'PGM_flux',
        'PIt2r_flux',
        'PPC_flux',
        'PPCK_flux',
        'PPS_flux',
        'PTAr_flux',
        'PYK_flux',
        'PYRt2_flux',
        'RPE_flux',
        'RPI_flux',
        'SUCCt2_2_flux',
        'SUCCt3_flux',
        'SUCDi_flux',
        'SUCOAS_flux',
        'TALA_flux',
        'THD2_flux',
        'TKT1_flux',
        'TKT2_flux',
        'TPI_flux'
    ]
    '''
    output_cols = [
        'EX_co2_e_flux', 'EX_h_e_flux', 'EX_h2o_e_flux',
        'EX_nh4_e_flux', 'EX_o2_e_flux', 'EX_pi_e_flux',
        'Biomass_Ecoli_core_flux'
    ]
    '''
    df = pd.read_csv(filename)

    # Fill missing inputs with 0 (i.e., not uptaken)
    df[input_cols] = df[input_cols].fillna(0)
    output_stds = df[output_cols].std()
    valid_outputs = output_stds[output_stds > 0.1].index.tolist()
    print(f"Original outputs: {len(output_cols)}, Valid outputs: {len(valid_outputs)}")

    print(f"\nLoaded data with {len(df)} samples from {filename}")

    X = df[input_cols].values.astype(np.float32)
    y = df[valid_outputs].values.astype(np.float32)

    return X, y, input_cols, valid_outputs

def run_cross_validation(X_train, y_train, k=5, epochs=300):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    r2_scores = []
    fold = 1

    for train_idx, val_idx in kf.split(X_train):
        print(f"\nFold {fold}/{k}:")
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Scale data per fold
        x_scaler_fold = StandardScaler().fit(X_train_fold)
        y_scaler_fold = StandardScaler().fit(y_train_fold)
        X_train_fold_scaled = x_scaler_fold.transform(X_train_fold)
        X_val_fold_scaled = x_scaler_fold.transform(X_val_fold)
        y_train_fold_scaled = y_scaler_fold.transform(y_train_fold)
        y_val_fold_scaled = y_scaler_fold.transform(y_val_fold)

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_fold_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_fold_scaled, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_fold_scaled, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_fold_scaled, dtype=torch.float32)

        # Initialize model
        model = MetabolicNN(
            input_size=X_train_fold.shape[1],
            output_size=y_train_fold_scaled.shape[1]
        )
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train model
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            preds = model(X_val_tensor).numpy()
            preds_unscaled = y_scaler_fold.inverse_transform(preds)
            true_unscaled = y_scaler_fold.inverse_transform(y_val_tensor.numpy())
            r2 = r2_score(true_unscaled, preds_unscaled)
            r2_scores.append(r2)
            print(f"R²: {r2:.4f}")
        
        fold += 1

    print(f"\nCross-Validation R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    return r2_scores

def identify_problematic_fluxes(y_train, threshold=0.01):
    problematic = []
    for i in range(y_train.shape[1]):
        zero_ratio = np.mean(np.abs(y_train[:, i]) < threshold)
        if zero_ratio > 0.3:  # More than 30% near-zero
            problematic.append(i)
    return problematic

def track_gradient_norms(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5
    
def plot_loss_curves(train_losses, test_losses, save_path, log_scale=True):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(14, 10))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(test_losses, label="Test Loss")
    if log_scale:
        plt.yscale('log')
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("MSE Loss", fontsize=18)
    plt.title("Training and Test Loss", fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.savefig(save_path)
    plt.close()
    #print(f"\nTraining curve saved to {save_path}")
    
def plot_diagnostics_2x2(y_true, y_pred, label, save_path):
    """Creates a 2x2 matrix of plots: true vs predicted, residuals, error distribution, and histogram of actuals"""
    residuals = y_true - y_pred

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Actual vs Predicted
    axs[0, 0].scatter(y_true, y_pred, alpha=0.2, s=5, color='royalblue')
    axs[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    axs[0, 0].set_title(f'True vs Predicted: {label}')
    axs[0, 0].set_xlabel('True value')
    axs[0, 0].set_ylabel('Predicted')
    axs[0, 0].grid(True)

    # Residuals plot
    axs[0, 1].scatter(y_true, residuals, alpha=0.15, color='darkorange')
    axs[0, 1].axhline(y=0, color='r', linestyle='-')
    axs[0, 1].set_title(f'Residuals: {label}')
    axs[0, 1].set_xlabel('True value')
    axs[0, 1].set_ylabel('Residuals')
    axs[0, 1].grid(True)

    # Error distribution
    sns.histplot(residuals, kde=True, ax=axs[1, 0], legend=False, color='indianred')
    axs[1, 0].set_title(f'Prediction Error Distribution: {label}')
    axs[1, 0].set_xlabel('Prediction Error')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].grid(True)

    # Histogram of actual values
    sns.histplot(y_true, kde=True, ax=axs[1, 1], legend=False, color='mediumseagreen')
    axs[1, 1].set_title(f'True Value Distribution: {label}')
    axs[1, 1].set_xlabel('True Value')
    axs[1, 1].set_ylabel('Frequency')
    axs[1, 1].grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def save_individual_diagnostic_plots(y_true, y_pred, label, save_path):
    """Save individual diagnostic plots separately"""
    residuals = y_true - y_pred

    # Actual vs Predicted
    plt.figure(figsize=(14, 10))
    plt.scatter(y_true, y_pred, alpha=0.2, s=10, color='royalblue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('True value', fontsize=18)
    plt.ylabel('Predicted', fontsize=18)
    plt.title(f'True vs Predicted: {label}', fontsize=20)
    plt.grid(True)
    plt.savefig(f"{save_path}_{label}_true_vs_pred.png")
    plt.close()

    # Residuals plot
    plt.figure(figsize=(14, 10))
    plt.scatter(y_true, residuals, alpha=0.2, color='darkorange')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('True value', fontsize=18)
    plt.ylabel('Residuals', fontsize=18)
    plt.title(f'Residuals: {label}', fontsize=20)
    plt.grid(True)
    plt.savefig(f"{save_path}_{label}_residuals.png")
    plt.close()

    # Error distribution
    plt.figure(figsize=(14, 10))
    sns.histplot(residuals, kde=True, color='indianred')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Prediction Error', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.title(f'Error Distribution: {label}', fontsize=20)
    plt.grid(True)
    plt.savefig(f"{save_path}_{label}_error_dist.png")
    plt.close()

    # Histogram of actual values
    plt.figure(figsize=(14, 10))
    sns.histplot(y_true, kde=True, color='mediumseagreen')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('True value', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.title(f'True Value Distribution: {label}', fontsize=20)
    plt.grid(True)
    plt.savefig(f"{save_path}_{label}_true_dist.png")
    plt.close()

def plot_feature_importance(model, feature_names, save_path):
    """Visualize feature importance using first-layer weights"""
    weights = model.model[0].weight.data.numpy()
    importance = np.mean(np.abs(weights), axis=0)
    
    plt.figure(figsize=(14, 13))
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=16)
    plt.bar(feature_names, importance, color='seagreen')
    plt.xlabel('Features', fontsize=18)
    plt.ylabel('Average Absolute Weight', fontsize=18)
    plt.title('Feature Importance from First Layer Weights', fontsize=20)
    plt.savefig(save_path)
    plt.close()

def plot_gradient_norms(gradient_norms, save_path=None, log_scale=True):
    plt.figure(figsize=(14, 10))
    plt.plot(gradient_norms)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Gradient Norm (L2)", fontsize=18)
    plt.title("Gradient Norms Over Epochs", fontsize=20)
    if log_scale:
        plt.yscale('log')
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        #print(f"Gradient norm plot saved to {save_path}")
    else:
        plt.show()

def plot_standardized_errors(y_true, y_pred, output_labels, y_scaler, save_path):
    """Plot standardized errors across all outputs"""
    plt.figure(figsize=(14, 10))
    
    for i, label in enumerate(output_labels):
        # Calculate standardized residuals (original units / std of training data)
        std = y_scaler.scale_[i]
        residuals = (y_true[:, i] - y_pred[:, i]) / std
        
        sns.kdeplot(residuals, label=f"{label} (σ={std:.2f})", fill=True, alpha=0.15)
    
    plt.axvline(0, color='r', linestyle='--')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Standardized Residuals (Original Units / σ)', fontsize=18)
    plt.ylabel('Density', fontsize=18)
    plt.title('Standardized Error Distributions Across Outputs', fontsize=20)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


X, y, input_cols, output_cols = load_and_preprocess_data(datafile)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

problematic_indices = identify_problematic_fluxes(y_train)
print(f"Problematic fluxes: {[output_cols[i] for i in problematic_indices]}")

#cv_scores = run_cross_validation(X_train, y_train, k=5, epochs=300)

# Train Final Model on entire training set
x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)
X_train_scaled = x_scaler.transform(X_train)
X_test_scaled = x_scaler.transform(X_test)
y_train_scaled = y_scaler.transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

model = MetabolicNN(
    input_size=X_train.shape[1],
    output_size=y_train_scaled.shape[1]
)
#criterion = nn.MSELoss()
criterion = nn.HuberLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

train_losses = []
test_losses = []
gradient_norms = []
today = date.today().isoformat()

print("\nTrain Final Model on entire training set:")
epochs = 500

best_test_loss = float('inf')
best_epoch = -1

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    grad_norm = track_gradient_norms(model)
    gradient_norms.append(grad_norm)
    optimizer.step()
    train_losses.append(loss.item())

    # Validation on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor).item()
        test_losses.append(test_loss)
    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_epoch = epoch

    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}")

print(f"Best test loss at epoch {best_epoch+1}: {best_test_loss:.4f}\n")

# Evaluate final model on test set
model.eval()
with torch.no_grad():
    test_preds_scaled = model(X_test_tensor).numpy()
    test_preds = y_scaler.inverse_transform(test_preds_scaled)
    test_true = y_scaler.inverse_transform(y_test_tensor.numpy())

pic_dir = f"./pics/{today}"
os.makedirs(pic_dir, exist_ok=True)
model_name = "ecoli_core_v4"

# Plot training curves
plot_loss_curves(train_losses, test_losses, f'{pic_dir}/{model_name}_training_curve.png')

for i, label in enumerate(output_cols):
    actual = test_true[:, i]
    predicted = test_preds[:, i]

    plot_diagnostics_2x2(actual, predicted,
                         label,
                         f'{pic_dir}/{model_name}_diagnostics_{label}.png')
    #save_individual_diagnostic_plots(actual, predicted, label, f'{pic_dir}/{model_name}')

#plot_feature_importance(model, input_cols, f'{pic_dir}/{model_name}_feature_importance.png')

plot_gradient_norms(gradient_norms, f"{pic_dir}/{model_name}_gradient_norms.png")

#plot_standardized_errors(test_true, test_preds, output_labels, y_scaler,
#                         f'{pic_dir}/{model_name}_standardized_errors.png')

for i, label in enumerate(output_cols):
    r2 = r2_score(test_true[:, i], test_preds[:, i])
    print(f"{label}: R² = {r2:.4f}")

torch.save(model.state_dict(), f"./models/{model_name}_metabolic_nn.pth")
import joblib
joblib.dump(x_scaler, f"./models/{model_name}_input_scaler.pkl")
joblib.dump(y_scaler, f"./models/{model_name}_output_scaler.pkl")

print("\nModel and scalers saved.")
