import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mpmath import mp
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import joblib

mp.dps = 60  # high precision decimal places

# ==== Chaotic systems sequence generation ==== #

def generate_logistic_sequence(x0, mu=3.99, length=1000):
    """Generate logistic map sequence in high precision and quantize to bits."""
    seq = []
    x = mp.mpf(x0)
    mu = mp.mpf(mu)
    for _ in range(length):
        x = mu * x * (1 - x)
        # Quantize x to a bit 0 or 1 based on threshold 0.5
        bit = 1 if x > 0.5 else 0
        seq.append(bit)
    return np.array(seq, dtype=np.uint8)

def generate_lorenz_sequence(initial_state, length=1000, dt=0.01):
    """Generate Lorenz system sequence and quantize x coordinate."""
    sigma = 10.0
    beta = 8/3
    rho = 28.0

    def lorenz(t, state):
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return [dx, dy, dz]

    t_span = (0, length*dt)
    t_eval = np.linspace(*t_span, length)

    sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval, method='RK45')

    # Normalize x coordinate to [0,1]
    x_vals = sol.y[0]
    x_norm = (x_vals - x_vals.min()) / (x_vals.max() - x_vals.min())

    # Quantize x_norm to bits by thresholding at 0.5
    seq = (x_norm > 0.5).astype(np.uint8)

    return seq

def generate_chen_sequence(initial_state, length=1000, dt=0.01):
    """Generate Chen system sequence and quantize x coordinate."""
    a = 35.0
    b = 3.0
    c = 28.0

    def chen(t, state):
        x, y, z = state
        dx = a * (y - x)
        dy = (c - a) * x - x * z + c * y
        dz = x * y - b * z
        return [dx, dy, dz]

    t_span = (0, length*dt)
    t_eval = np.linspace(*t_span, length)

    sol = solve_ivp(chen, t_span, initial_state, t_eval=t_eval, method='RK45')

    x_vals = sol.y[0]
    x_norm = (x_vals - x_vals.min()) / (x_vals.max() - x_vals.min())

    seq = (x_norm > 0.5).astype(np.uint8)
    return seq

# ==== GRU Model ==== #

class ChaoticGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden=None):
        out, hidden = self.gru(x, hidden)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out, hidden

# ==== Data preparation ==== #

def prepare_training_data(seq_length=1000, sample_length=100):
    """
    Generate mixed chaotic sequences and prepare train/test datasets.
    Labels: 0=Logistic,1=Lorenz,2=Chen (for info, not used in training output)
    """

    # Generate sequences
    logistic_seq = generate_logistic_sequence(0.1, length=seq_length)
    lorenz_seq = generate_lorenz_sequence([0.1, 0.0, 0.0], length=seq_length)
    chen_seq = generate_chen_sequence([0.1, 0.0, 0.0], length=seq_length)

    # Combine with labels
    sequences = [logistic_seq, lorenz_seq, chen_seq]
    labels = [0, 1, 2]

    X = []
    y = []

    # For each system, create samples of length sample_length (sequence prediction)
    for seq in sequences:
        for i in range(len(seq) - sample_length):
            # Input sequence: bits from i to i+sample_length-1
            # Target: next bit (i+sample_length)
            X.append(seq[i:i+sample_length])
            y.append(seq[i+sample_length])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Reshape for GRU input: (samples, seq_len, features=1)
    X = np.expand_dims(X, axis=2)

    # Shuffle and split
    X, y = shuffle(X, y, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# ==== Training function ==== #

def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=128, lr=0.005):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs, _ = model(inputs)
            outputs = outputs[:, -1, :]  # last output for prediction
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # Evaluate
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).unsqueeze(1)
                outputs, _ = model(inputs)
                outputs = outputs[:, -1, :]
                preds = (outputs > 0.5).float()
                total += targets.size(0)
                correct += (preds == targets).sum().item()
        accuracy = correct / total

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Test Accuracy={accuracy*100:.2f}%")

    return model

# ==== Generate keystream bits using trained model ==== #

def generate_keystream_from_gru(model, seed_bits, length):
    """
    Generate keystream bits by feeding seed_bits as input and iteratively
    predicting next bits.
    """
    model.eval()
    device = next(model.parameters()).device

    input_seq = torch.tensor(seed_bits, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(device)
    generated_bits = []

    hidden = None
    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_seq, hidden)
            next_bit_prob = output[:, -1, 0].item()
            next_bit = 1 if next_bit_prob > 0.5 else 0
            generated_bits.append(next_bit)

            # Append predicted bit and remove first bit to keep length
            next_input = torch.tensor([[[next_bit]]], dtype=torch.float32).to(device)
            input_seq = torch.cat((input_seq[:, 1:, :], next_input), dim=1)

    return np.array(generated_bits, dtype=np.uint8)

# ==== Main ==== #

def main():
    # 1) Train model from mixed chaotic data
    X_train, X_test, y_train, y_test = prepare_training_data(seq_length=2000, sample_length=50)
    model = ChaoticGRU()
    model = train_model(model, X_train, y_train, X_test, y_test, epochs=20)

    # Save model
    torch.save(model.state_dict(), "gru_chaotic_model.pth")
    print("Saved GRU model to gru_chaotic_model.pth")

    # 2) Generate keystream from final key bits as seed
    # Here we simulate final key bits as random bits for demonstration:
    final_key_bits = np.random.randint(0, 2, 50)  # seed length = sample_length

    # Load model and generate keystream
    model.load_state_dict(torch.load("gru_chaotic_model.pth"))
    model.eval()

    keystream_length = 512
    keystream = generate_keystream_from_gru(model, final_key_bits, keystream_length)
    print(f"Generated keystream (length {len(keystream)} bits):")
    print(keystream)

    # Save keystream for later use
    np.save("phase6b_keystream.npy", keystream)
    print("Keystream saved to phase6b_keystream.npy")

if __name__ == "__main__":
    main()
