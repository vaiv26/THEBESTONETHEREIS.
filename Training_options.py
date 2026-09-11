import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------


def assess_device():
    """Return the best available device: MPS, CUDA, then CPU."""

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------

def embedding_size(cardinality):
    return min(50, max (2,(cardinality // 2)))

def as_category_text(series):
    return series.astype("string").fillna("Missing").astype(str)

# 1 MLP
class TabularSalesModel_MLP(nn.Module):
    def __init__(self, embedding_cardinalities, embedding_dimensions, n_numeric_features, dropout):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=cardinality, embedding_dim=dimension)
                for cardinality, dimension in zip(embedding_cardinalities, embedding_dimensions)
            ]
        )
        input_size = sum(embedding_dimensions) + n_numeric_features
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, categorical_inputs, numeric_inputs):
        embedded_features = [
            embedding_layer(categorical_inputs[:, index])
            for index, embedding_layer in enumerate(self.embeddings)
        ]
        x = torch.cat(embedded_features + [numeric_inputs], dim=1)
        return self.network(x)

#2 RNN, LSTM, GRU
class RecurrentSalesModel(nn.Module):
    def __init__(self, model_type, encoded_feature_names, hiddenSize, numLayers, dropout, storeEmbeddingSize,
                 selected_stores):
        super().__init__()
        recurrent_class = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[model_type]
        self.recurrent = recurrent_class(
            input_size=len(encoded_feature_names) + 1,
            hidden_size=hiddenSize,
            num_layers=numLayers,
            batch_first=True,
            dropout=dropout if numLayers > 1 else 0.0,
        )
        self.store_embedding = nn.Embedding(len(selected_stores), storeEmbeddingSize)
        self.head = nn.Sequential(
            nn.Linear(hiddenSize + storeEmbeddingSize + len(encoded_feature_names) + 1, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, history, future_features, store_ids, horizon):
        _, hidden = self.recurrent(history)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        context = torch.cat([hidden[-1], self.store_embedding(store_ids)], dim=1)
        batch_size, steps, _ = future_features.shape
        context = context.unsqueeze(1).expand(-1, steps, -1)
        lead = torch.arange(1, steps + 1, device=history.device, dtype=history.dtype) / horizon
        lead = lead.view(1, steps, 1).expand(batch_size, -1, -1)
        combined = torch.cat([context, future_features, lead], dim=-1)
        return self.head(combined).squeeze(-1)

# 3 Transformers
class TransformerSalesModel(nn.Module):
    def __init__(
        self,
        hidden_size,
        nhead,
        num_layers,
        dropout,
        ff_multiplier,
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            len(encoded_feature_names) + 1,
            hidden_size
        )

        self.positional_encoding = nn.Parameter(
            torch.zeros(1, LOOKBACK, hidden_size)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * ff_multiplier,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.store_embedding = nn.Embedding(
            len(selected_stores),
            STORE_EMBEDDING_SIZE
        )

        self.head = nn.Sequential(
            nn.Linear(
                hidden_size
                + STORE_EMBEDDING_SIZE
                + len(encoded_feature_names)
                + 1,
                64,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, history, future_features, store_ids):

        x = self.input_projection(history)

        x = x + self.positional_encoding[:, :x.size(1), :]

        x = self.transformer(x)

        context = x[:, -1, :]

        store_context = self.store_embedding(store_ids)

        context = torch.cat(
            [context, store_context],
            dim=1
        )

        batch_size, steps, _ = future_features.shape

        context = context.unsqueeze(1).expand(
            -1, steps, -1
        )

        lead = (
            torch.arange(
                1,
                steps + 1,
                device=history.device,
                dtype=history.dtype
            )
            / HORIZON
        )

        lead = lead.view(1, steps, 1).expand(
            batch_size, -1, -1
        )

        combined = torch.cat(
            [context, future_features, lead],
            dim=-1
        )

        return self.head(combined).squeeze(-1)

# ---------------------------------------------------------------------
# Sliding window and make forecast and loaders
# ---------------------------------------------------------------------

class SalesWindowDataset(Dataset):
    def __init__(self, frame, lookback, horizon, stride, store_to_index, numeric_features, numeric_means, category_levels, numeric_stds, sales_mean, sales_std):
        self.lookback, self.horizon = lookback, horizon
        self.series, self.windows = {}, []
        self.skipped_windows = 0
        for store, group in frame.groupby("Store", sort=True):
            group = group.sort_values("Date")
            features = encode_features(group, numeric_features, numeric_means, category_levels, numeric_stds)
            target = encode_sales(group["Sales"], sales_mean, sales_std)
            history = np.column_stack([target, features]).astype(np.float32)
            dates = group["Date"].to_numpy(dtype="datetime64[D]")
            store_id = store_to_index[int(store)]
            self.series[store_id] = (history, features, target, dates)
            last_start = len(group) - horizon
            starts = list(range(lookback, last_start + 1, stride))
            if last_start >= lookback and last_start not in starts:
                starts.append(last_start)
            for start in starts:
                dates_in_window = dates[start - lookback:start + horizon]
                if np.all(np.diff(dates_in_window) == np.timedelta64(1, "D")):
                    self.windows.append((store_id, start))
                else:
                    self.skipped_windows += 1
        if not self.windows:
            raise ValueError("No complete training windows. Use more history or a shorter LOOKBACK.")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        store_id, start = self.windows[index]
        history, features, target, _ = self.series[store_id]
        return (
            torch.from_numpy(history[start - self.lookback:start]),
            torch.from_numpy(features[start:start + self.horizon]),
            torch.tensor(store_id, dtype=torch.long),
            torch.from_numpy(target[start:start + self.horizon]),
        )


def make_train_loader(train_dataset, batch_size, random_state=42):
    return DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(random_state),
    )

def make_forecast_inputs(history_frame, future_frame, horizon, lookback, store_to_index,
                         numeric_features, numeric_means, category_levels, numeric_stds,
                         sales_mean, sales_std):
    forbidden = {"Sales", "Customers"} & set(future_frame.columns)
    if forbidden:
        raise ValueError(f"Future features must exclude: {sorted(forbidden)}")
    history_groups = {
        int(store): group.sort_values("Date")
        for store, group in history_frame.groupby("Store")
    }
    histories, futures, store_ids, rows = [], [], [], []
    common_dates = pd.date_range(future_frame["Date"].min(), future_frame["Date"].max(), freq="D")
    if not 1 <= len(common_dates) <= horizon:
        raise ValueError("Future date span must be between 1 and HORIZON days.")
    for store, group in future_frame.groupby("Store", sort=True):
        group = group.sort_values("Date")
        assert np.array_equal(group["Date"].to_numpy(), common_dates.to_numpy())
        origin = common_dates[0]
        past = history_groups[int(store)]
        past = past[past["Date"] < origin].tail(lookback)
        expected_past = pd.date_range(origin - pd.Timedelta(days=lookback), periods=lookback)
        assert np.array_equal(past["Date"].to_numpy(), expected_past.to_numpy()), (
            f"Store {store} does not have {lookback} consecutive historical days."
        )
        histories.append(np.column_stack([encode_sales(past["Sales"], sales_mean, sales_std), encode_features(past, numeric_features, numeric_means, category_levels, numeric_stds)]))
        futures.append(encode_features(group, numeric_features, numeric_means, category_levels, numeric_stds))
        store_ids.append(store_to_index[int(store)])
        rows.append(group)
    tensors = (
        torch.tensor(np.stack(histories), dtype=torch.float32),
        torch.tensor(np.stack(futures), dtype=torch.float32),
        torch.tensor(store_ids, dtype=torch.long),
    )
    forecast_rows = pd.concat(rows, ignore_index=True)
    return tensors, forecast_rows

# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

# Data Loader for MLP
def encode_categorical_frame(df, categorical_features, category_maps):
    encoded = pd.DataFrame(index=df.index)
    for column in categorical_features:
        mapping = category_maps[column]
        encoded[column] = as_category_text(df[column]).map(mapping).fillna(0).astype("int64")
    return encoded

def scale_numeric_frame(df, numeric_features, numeric_means, numeric_stds):
    numeric = df[numeric_features].astype(float).copy()
    numeric = numeric.fillna(numeric_means)
    numeric = (numeric - numeric_means) / numeric_stds
    return numeric.astype("float32")

class StoreSalesDataset(Dataset):
    def __init__(self, categorical_data, numeric_data, target=None):
        self.categorical_data = torch.tensor(categorical_data.values, dtype=torch.long)
        self.numeric_data = torch.tensor(numeric_data.values, dtype=torch.float32)
        self.target = None if target is None else torch.tensor(target.values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.numeric_data)

    def __getitem__(self, index):
        if self.target is None:
            return self.categorical_data[index], self.numeric_data[index]
        return self.categorical_data[index], self.numeric_data[index], self.target[index]
# MLP data loader end


def make_dataloader(dataset, batch_size=64, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def stores_covering_dates(frame, dates):
    counts = frame.loc[frame["Date"].isin(dates)].groupby("Store")["Date"].nunique()
    return set(counts[counts == len(dates)].index)

def encode_features(frame, numeric_features, numeric_means, category_levels, numeric_stds):
    numeric = frame[numeric_features].astype(float).fillna(numeric_means)
    blocks = [((numeric - numeric_means) / numeric_stds).to_numpy(dtype=np.float32)]
    for col, levels in category_levels.items():
        values = frame[col].fillna("missing").astype(str)
        blocks.append(np.column_stack(
            [values.eq(level).to_numpy(dtype=np.float32) for level in levels]
            + [(~values.isin(levels)).to_numpy(dtype=np.float32)]
        ))
    encoded = np.concatenate(blocks, axis=1).astype(np.float32)
    assert np.isfinite(encoded).all(), "Features contain non-finite values."
    return encoded

def encode_sales(values, sales_mean, sales_std):
    return ((np.log1p(np.asarray(values, dtype=np.float64)) - sales_mean) / sales_std).astype(np.float32)

def decode_sales(values, open_values, sales_std, sales_mean):
    values = np.asarray(values, dtype=np.float64)
    sales = np.maximum(np.expm1(values * sales_std + sales_mean), 0.0)
    sales = np.where(np.asarray(open_values) == 0, 0.0, sales)
    assert np.isfinite(sales).all(), "Predictions diverged. Try a lower learning rate."
    return sales


def read_split(
    test="",
    train ="",
    valid ="",
    device="cpu",
    train_size=0.70,
    Lookback= 28,
    MAX_STORES = None,
    HISTORY_START = None,
):

    numeric_features = [
    "Open",
    "Promo",
    "SchoolHoliday",
    "CompetitionDistance",
    "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear",
    "Promo2",
    "Promo2SinceWeek",
    "Promo2SinceYear",
    "year",
    "month",
    "day",
    "weekofyear",
    "quarter",
    "dayofyear",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "days_since_start",
    "competition_open_months",
    "promo2_active",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    ]

    categorical_features = ["StateHoliday", "StoreType", "Assortment", "DayOfWeek"]
    feature_columns = numeric_features + categorical_features
    base_columns = ["Store", "Date"] + feature_columns


    csv_options = {"parse_dates": ["Date"], "dtype": {
    column: "string" for column in categorical_features
    }}

    val_all = pd.read_csv( valid, usecols=base_columns + ["Sales"], **csv_options)

    test_all = pd.read_csv(test, usecols=base_columns + ["Id"], **csv_options)

    train_keys = pd.read_csv(train, usecols=["Store", "Date"], parse_dates=["Date"])

    assert train_keys["Date"].max() < val_all["Date"].min()
    assert val_all["Date"].max() < test_all["Date"].min()
    assert not train_keys.duplicated(["Store", "Date"]).any()
    assert not val_all.duplicated(["Store", "Date"]).any()
    assert not test_all.duplicated(["Store", "Date"]).any()

    val_dates = pd.date_range(val_all["Date"].min(), val_all["Date"].max(), freq="D")
    test_dates = pd.date_range(test_all["Date"].min(), test_all["Date"].max(), freq="D")

    HORIZON = max(len(val_dates), len(test_dates))

    val_history_dates = pd.date_range(
        val_dates[0] - pd.Timedelta(days=Lookback), periods=Lookback, freq="D"
    )
    test_history_dates = pd.date_range(
        test_dates[0] - pd.Timedelta(days=Lookback), periods=Lookback, freq="D"
    )

    all_history_keys = pd.concat([train_keys, val_all[["Store", "Date"]]], ignore_index=True)
    eligible = sorted(
        stores_covering_dates(train_keys, val_history_dates)
        & stores_covering_dates(val_all, val_dates)
        & stores_covering_dates(all_history_keys, test_history_dates)
        & stores_covering_dates(test_all, test_dates)
    )

    selected_stores = eligible 

    # Read only required columns and retain the selected stores chunk by chunk.
    parts = []
    for chunk in pd.read_csv(
        train, usecols=base_columns + ["Sales"],
        chunksize=100_000, **csv_options
    ):
        keep = chunk["Store"].isin(selected_stores)
        if HISTORY_START is not None:
            keep &= chunk["Date"] >= pd.Timestamp(HISTORY_START)
        parts.append(chunk.loc[keep])
    train_df = pd.concat(parts, ignore_index=True).sort_values(["Store", "Date"]).reset_index(drop=True)
    val_df = val_all[val_all["Store"].isin(selected_stores)].sort_values(
        ["Store", "Date"]
    ).reset_index(drop=True)
    test_df = test_all[test_all["Store"].isin(selected_stores)].sort_values(
        ["Store", "Date"]
    ).reset_index(drop=True)
    full_prepared_test_rows = len(test_all)
    del parts, train_keys, all_history_keys

    for frame in (train_df, val_df):
        assert np.isfinite(frame["Sales"]).all() and frame["Sales"].ge(0).all()
    for frame in (train_df, val_df, test_df):
        assert frame["Open"].isin([0, 1]).all()

    return {
        "traindf": train_df,
        "valdf": val_df,
        "testdf": test_df,
        "selectedStores": selected_stores,
        "eligibleStores": eligible,
        "Horizon" : HORIZON,
        "valDates":val_dates,
        "testDates": test_dates,
        "fullPreparedtestRows":full_prepared_test_rows,
        "categoricalFeatures" : categorical_features,
        "numericalFeatures" : numeric_features,
    }

"""
    # The only categorical feature is sex.  We turn it into dummy variables.
    df = pd.get_dummies(df, columns=["sex"], dtype=float)

    X_df = df.drop("rings", axis=1)
    y_np = df["rings"].to_numpy(dtype=np.float32).reshape(-1, 1)

    X_train_np, X_temp_np, y_train_np, y_temp_np = train_test_split(
        X_df.to_numpy(dtype=np.float32),
        y_np,
        train_size=train_size,
        random_state=random_state,
    )

    X_val_np, X_test_np, y_val_np, y_test_np = train_test_split(
        X_temp_np,
        y_temp_np,
        test_size=0.50,
        random_state=random_state,
    )

    # Fit the scaler only on training data, then apply to validation and test.
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train_np).astype(np.float32)
    X_val_np = scaler.transform(X_val_np).astype(np.float32)
    X_test_np = scaler.transform(X_test_np).astype(np.float32)

    x_train = torch.tensor(X_train_np, device=device, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, device=device, dtype=torch.float32)
    x_val = torch.tensor(X_val_np, device=device, dtype=torch.float32)
    y_val = torch.tensor(y_val_np, device=device, dtype=torch.float32)
    x_test = torch.tensor(X_test_np, device=device, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, device=device, dtype=torch.float32)

    train_loader = make_dataloader(x_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_dataloader(x_val, y_val, batch_size=len(y_val), shuffle=False)
    test_loader = make_dataloader(x_test, y_test, batch_size=len(y_test), shuffle=False)

    return {
        "df": df,
        "feature_columns": list(X_df.columns),
        "scaler": scaler,
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "batch_size": batch_size,
        "num_feats": x_train.shape[1],
    }
"""


# ---------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------


def create_scheduler(optimizer, step_size=20, gamma=0.7):
    """Create a StepLR scheduler."""

    return StepLR(optimizer, step_size=step_size, gamma=gamma)


# ---------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------


def update_early_stopping(
    model,
    val_loss,
    state=None,
    patience=4,
    min_delta=0.001,
    restore_best_weights=True,
):
    """Update early-stopping state and return (should_stop, state)."""

    val_loss = float(val_loss)

    if state is None:
        state = {
            "best_loss": val_loss,
            "best_model": copy.deepcopy(model.state_dict()),
            "counter": 0,
            "status": "First validation loss saved.",
        }
        return False, state

    if state["best_loss"] - val_loss >= min_delta:
        state["best_loss"] = val_loss
        state["best_model"] = copy.deepcopy(model.state_dict())
        state["counter"] = 0
        state["status"] = "Improvement found, counter reset to 0."
        return False, state

    state["counter"] += 1
    state["status"] = f"No improvement in the last {state['counter']} epochs."

    if state["counter"] >= patience:
        state["status"] = f"Early stopping triggered after {state['counter']} epochs."
        if restore_best_weights and state["best_model"] is not None:
            model.load_state_dict(state["best_model"])
        return True, state

    return False, state


# ---------------------------------------------------------------------
# l1 normalization
# ---------------------------------------------------------------------


def l1_penalty(model):
    """Return the L1 norm of all model parameters."""

    return sum(parameter.abs().sum() for parameter in model.parameters())

# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def mlp_basic_training_loop(
    model,
    train_loader,
    loss_function,
    optimizer,
    epochs=50,
    val_loader=None,
    scheduler=None,
    l1_lambda=0.0,
    patience=None,
    min_delta=0.001,
    print_every=2,
    model_path = "",
    device = any
):
    """Train any regression model with the same 5-step PyTorch loop."""

    history = {"train_loss": [], "valid_loss": [], "valid_mae": []}
    early_state = None
    best_epoch = epochs
    checkpoint_path = model_path

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for categorical_inputs, numerical_inputs, y in train_loader:
            categorical_inputs = categorical_inputs.to(device)
            numerical_inputs = numerical_inputs.to(device)
            y = y.to(device)

            # Forward pass
            preds = model(categorical_inputs, numerical_inputs)

            # 2. Calculate loss
            base_loss = loss_function(preds, y)
            loss = base_loss + l1_lambda * l1_penalty(model)

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backwards
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

            batch_losses.append(base_loss.item())
            
        if scheduler is not None:
            scheduler.step()
        history["train_loss"].append(float(np.mean(batch_losses)))

        if val_loader is not None:
            model.eval()
            valid_loss_total = 0.0
            valid_abs_error_total = 0.0
            valid_observations = 0

            with torch.inference_mode():
                for categorical_inputs, numerical_inputs, y_valid in val_loader:
                    valid_preds = model(categorical_inputs, numerical_inputs)
                    valid_loss = loss_function(valid_preds, y_valid)
                    batch_size = y_valid.shape[0]

                    valid_loss_total += valid_loss.item() * batch_size
                    valid_abs_error_total += torch.abs(valid_preds - y_valid).sum().item()
                    valid_observations += batch_size

            valid_loss = valid_loss_total / valid_observations
            valid_mae = valid_abs_error_total / valid_observations

            history["valid_loss"].append(valid_loss)
            history["valid_mae"].append(valid_mae)

            if valid_loss == min(history["valid_loss"]):
                best_epoch = epoch + 1

            if print_every and (epoch + 1) % print_every == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Train Loss: {history['train_loss'][-1]:.4f}, "
                    f"Validation Loss: {valid_loss:.4f}, "
                    f"Validation MAE: {valid_mae:.4f}"
                )

            if patience is not None:
                stop, early_state = update_early_stopping(
                    model,
                    valid_loss,
                    state=early_state,
                    patience=patience,
                    min_delta=min_delta,
                )
                if stop:
                    print(f"Stopping at epoch {epoch + 1}. {early_state['status']}")
                    break

    return model, history, best_epoch


def basic_training_loop(
    model,
    train_loader,
    loss_function,
    optimizer,
    epochs=50,
    val_loader=None,
    scheduler=None,
    l1_lambda=0.0,
    patience=None,
    min_delta=0.001,
    print_every=2,
    model_path = ""
):
    """Train any regression model with the same 5-step PyTorch loop."""

    history = {"train_loss": [], "valid_loss": [], "valid_mae": []}
    early_state = None
    best_epoch = epochs
    checkpoint_path = model_path

    for epoch in range(epochs):
        model.train()
        batch_losses = []

        for X, y in train_loader:
            # 1. Forward pass
            preds = model(X)

            # 2. Calculate loss
            base_loss = loss_function(preds, y)
            loss = base_loss + l1_lambda * l1_penalty(model)

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backwards
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

            batch_losses.append(base_loss.item())

        if scheduler is not None:
            scheduler.step()
        history["train_loss"].append(float(np.mean(batch_losses)))

        if val_loader is not None:
            model.eval()
            valid_loss_total = 0.0
            valid_abs_error_total = 0.0
            valid_observations = 0

            with torch.inference_mode():
                for X_valid, y_valid in val_loader:
                    valid_preds = model(X_valid)
                    valid_loss = loss_function(valid_preds, y_valid)
                    batch_size = y_valid.shape[0]

                    valid_loss_total += valid_loss.item() * batch_size
                    valid_abs_error_total += torch.abs(valid_preds - y_valid).sum().item()
                    valid_observations += batch_size

            valid_loss = valid_loss_total / valid_observations
            valid_mae = valid_abs_error_total / valid_observations

            history["valid_loss"].append(valid_loss)
            history["valid_mae"].append(valid_mae)

            if valid_loss == min(history["valid_loss"]):
                best_epoch = epoch + 1

            if print_every and (epoch + 1) % print_every == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Train Loss: {history['train_loss'][-1]:.4f}, "
                    f"Validation Loss: {valid_loss:.4f}, "
                    f"Validation MAE: {valid_mae:.4f}"
                )

            if patience is not None:
                stop, early_state = update_early_stopping(
                    model,
                    valid_loss,
                    state=early_state,
                    patience=patience,
                    min_delta=min_delta,
                )
                if stop:
                    print(f"Stopping at epoch {epoch + 1}. {early_state['status']}")
                    break

    return model, history, best_epoch

# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def regression_metrics(y_true, y_pred):
    """Compute standard regression metrics."""

    mse = torch.mean((y_pred - y_true) ** 2).item()
    rmse = float(np.sqrt(mse))
    mae = torch.mean(torch.abs(y_pred - y_true)).item()
    r2 = r2_score(
        y_true.detach().cpu().numpy().reshape(-1),
        y_pred.detach().cpu().numpy().reshape(-1),
    )
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def evaluate_regression(model, X, y, title="Evaluation", n_examples=10, print_results=True):
    """Evaluate a regression model and return metrics plus a prediction table."""

    model.eval()

    with torch.inference_mode():
        preds = model(X)

    metrics = regression_metrics(y, preds)
    prediction_table = pd.DataFrame(
        {
            "actual_rings": y.detach().cpu().numpy().reshape(-1)[:n_examples],
            "predicted_rings": np.round(
                preds.detach().cpu().numpy().reshape(-1)[:n_examples],
                2,
            ),
        }
    )

    if print_results:
        print(title)
        print(f"MSE:  {metrics['mse']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f} rings")
        print(f"MAE:  {metrics['mae']:.4f} rings")
        print(f"R^2:  {metrics['r2']:.4f}")

    return metrics, prediction_table


# ---------------------------------------------------------------------
# Bayesian optimization
# ---------------------------------------------------------------------


def choose_batch_size(batch_size):
    """Map a continuous suggestion to a simple batch-size choice."""

    allowed_batch_sizes = np.array([32, 64, 128])
    closest = np.argmin(np.abs(allowed_batch_sizes - batch_size))
    return int(allowed_batch_sizes[closest])


def make_hidden_sizes(neuron_pct, neuron_shrink, max_neurons=256, max_layers=4):
    """Build hidden-layer sizes from neuron percentage and shrink factor."""

    total_neurons = max(8, int(max_neurons * neuron_pct))
    relative_sizes = np.array([neuron_shrink**i for i in range(max_layers)])
    hidden_sizes = np.round(total_neurons * relative_sizes / relative_sizes.sum()).astype(int)
    return [max(4, int(size)) for size in hidden_sizes]


def make_bayesian_model(model_choice, num_feats, dropout, neuron_pct, neuron_shrink, device):
    """Build either a flexible Bayesian model or one of the fixed variants."""

    if isinstance(model_choice, str):
        hidden_sizes = make_hidden_sizes(neuron_pct, neuron_shrink)
        dropout_probability = dropout if model_choice == "bayes_dropout" else 0.0
        return AbaloneNetBayesian(num_feats, hidden_sizes, dropout_probability).to(device)

    return model_choice(num_feats).to(device)


def bayesian_objective_factory(
    model_choice,
    num_feats,
    x_train,
    y_train,
    x_val,
    y_val,
    device,
    max_epochs=80,
    random_state=42,
):
    """Create the objective function used by BayesianOptimization."""

    trials = []

    def objective(dropout, neuron_pct, neuron_shrink, log_lr, log_l1, log_l2, batch_size):
        params = {
            "dropout": dropout,
            "neuron_pct": neuron_pct,
            "neuron_shrink": neuron_shrink,
            "lr": 10**log_lr,
            "l1_lambda": 10**log_l1,
            "l2_lambda": 10**log_l2,
            "batch_size": choose_batch_size(batch_size),
        }

        torch.manual_seed(random_state + len(trials))
        model = make_bayesian_model(
            model_choice,
            num_feats,
            dropout=params["dropout"],
            neuron_pct=params["neuron_pct"],
            neuron_shrink=params["neuron_shrink"],
            device=device,
        )
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["l2_lambda"])
        scheduler = create_scheduler(optimizer, step_size=25, gamma=0.7)
        train_loader = make_dataloader(
            x_train,
            y_train,
            batch_size=params["batch_size"],
            shuffle=True,
        )
        val_loader = make_dataloader(
            x_val,
            y_val,
            batch_size=len(y_val),
            shuffle=False,
        )

        model, history, best_epoch = basic_training_loop(
            model,
            train_loader,
            loss_function,
            optimizer,
            epochs=max_epochs,
            val_loader=val_loader,
            scheduler=scheduler,
            l1_lambda=params["l1_lambda"],
            patience=12,
            print_every=None,
        )

        metrics, _ = evaluate_regression(
            model,
            x_val,
            y_val,
            title="Bayesian validation",
            print_results=False,
        )
        params["hidden_sizes"] = make_hidden_sizes(neuron_pct, neuron_shrink)
        params["best_epoch"] = best_epoch
        params["val_rmse"] = metrics["rmse"]
        trials.append(params)

        # bayesian-optimization maximizes, so return negative RMSE.
        return -metrics["rmse"]

    return objective, trials


def run_bayesian_optimization(objective, pbounds=None, init_points=5, n_iter=10, random_state=42):
    """Run BayesianOptimization with simple default bounds."""

    try:
        from bayes_opt import BayesianOptimization
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Install the package with: pip install bayesian-optimization"
        ) from error

    if pbounds is None:
        pbounds = {
            "dropout": (0.0, 0.4),
            "neuron_pct": (0.2, 1.0),
            "neuron_shrink": (0.3, 0.9),
            "log_lr": (-4, -2),
            "log_l1": (-8, -3),
            "log_l2": (-8, -3),
            "batch_size": (32, 128),
        }

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=random_state,
        verbose=2,
    )
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    return optimizer


def train_final_bayesian_model(
    model_choice,
    best_params,
    num_feats,
    x_dev,
    y_dev,
    device,
    epochs,
):
    """Train the selected Bayesian model on train+validation data."""

    model = make_bayesian_model(
        model_choice,
        num_feats,
        dropout=best_params["dropout"],
        neuron_pct=best_params["neuron_pct"],
        neuron_shrink=best_params["neuron_shrink"],
        device=device,
    )
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params["l2_lambda"],
    )
    scheduler = create_scheduler(optimizer, step_size=25, gamma=0.7)
    train_loader = make_dataloader(
        x_dev,
        y_dev,
        batch_size=int(best_params["batch_size"]),
        shuffle=True,
    )

    model, history, best_epoch = basic_training_loop(
        model,
        train_loader,
        loss_function,
        optimizer,
        epochs=epochs,
        scheduler=scheduler,
        l1_lambda=best_params["l1_lambda"],
        print_every=None,
    )
    return model, history, best_epoch
