import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class WaferLSTM(nn.Module):
    """
    Zaman serisi sensor akislari icin PyTorch tabanli LSTM (Long Short-Term Memory) sinir agi.
    """
    def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.2):
        super(WaferLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x boyutu: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # Sadece zaman penceresinin en son adimindaki (t) hafiza ciktisini al
        last_step_out = lstm_out[:, -1, :]
        out = self.dropout(last_step_out)
        out = self.sigmoid(self.fc(out))
        return out


def create_sequences(X_scaled, y_labels=None, seq_len=5):
    """
    2D sensor tablosunu LSTM'in okuyabilecegi 3D zaman pencerelerine (sequences) donusturur.
    """
    sequences = []
    targets = []
    for i in range(len(X_scaled) - seq_len + 1):
        sequences.append(X_scaled[i : i + seq_len])
        if y_labels is not None:
            targets.append(y_labels[i + seq_len - 1])
    X_seq = np.array(sequences, dtype=np.float32)
    if y_labels is not None:
        return X_seq, np.array(targets, dtype=np.float32)
    return X_seq


def train_and_predict_lstm(X_df, seq_len=5, epochs=25, lr=0.005):
    """
    Verilen sensor batch'i uzerinde hizli bir LSTM modeli egitir, egitim kayip (Loss) egrisini
    ve zaman serisi ariza olasiliklarini dondurur.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Eksik degerleri doldur ve olceklendir (Neural Network'ler olcekli veri ister)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_clean = scaler.fit_transform(imputer.fit_transform(X_df))

    # 2. Egitim icin sentetik/veya mevcut risk sinyali olustur (Zaman icinde biriken anomali)
    rolling_energy = np.mean(np.abs(X_clean), axis=1)
    y_pseudo = (rolling_energy > np.percentile(rolling_energy, 75)).astype(np.float32)

    # 3. 3 Boyutlu zaman pencerelerini olustur
    X_seq, y_seq = create_sequences(X_clean, y_pseudo, seq_len=seq_len)

    if len(X_seq) < 5:
        raise ValueError("LSTM zaman penceresi icin yeterli satir sayisi yok.")

    X_tensor = torch.tensor(X_seq)
    y_tensor = torch.tensor(y_seq).unsqueeze(1)

    # 4. Modeli, kayip fonksiyonunu (Binary Cross Entropy) ve optimize ediciyi (Adam) baslat
    model = WaferLSTM(input_size=X_df.shape[1], hidden_size=32)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    # 5. Tahminleri al
    model.eval()
    with torch.no_grad():
        final_probs = model(X_tensor).squeeze().numpy()

    return loss_history, final_probs
