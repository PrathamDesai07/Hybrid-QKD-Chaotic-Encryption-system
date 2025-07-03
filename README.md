# 🔐 Hybrid-QKD-Chaotic-Encryption-System
A complete multi-phase simulation of a hybrid quantum-classical encryption pipeline. This project integrates Quantum Key Distribution (QKD) with chaotic maps, machine learning, GRU-based keystream generation, and classical cryptographic techniques.

---

## 📌 Overview
This system simulates all phases from quantum key generation to final encrypted communication.

### 🧩 Phases:
| Phase | Description                                      |
|-------|--------------------------------------------------|
| 1     | BB84-based Quantum Key Distribution (QKD)        |
| 2     | QBER Estimation                                  |
| 3     | Chaotic Map-based Keystream Generation           |
| 4     | ML-based Chaotic Map Selection                   |
| 5     | GRU-based Stream Extension                       |
| 6     | Multi-layer Classical Encryption                 |
| 7     | Unified Simulation & Attack Resistance Evaluation|
| 8     | Final Report & Documentation                     |

---

## 🗂️ Repository Structure
```
Hybrid-QKD-Chaotic-Encryption-System/
├── main.py                          # Full Phase-7 integration
├── bb84_pipeline.py                 # BB84 QKD core logic
├── classical_encryption_engine.py   # Custom symmetric cipher engine
├── feedback_chaos.py                # Feedback-chaotic stream generator
├── gru_keystream.py                 # GRU-based keystream
├── ml_map_selection.py              # ML-based chaotic map selector
├── virtual_noise_layer.py           # Noise masking & unmasking
├── final_key.npy                    # Shared secret key
├── gru_chaotic_model.pth            # Saved PyTorch GRU model
├── attack_qber.png                  # Bar graph of attack QBERs
├── qber_vs_loss.png                 # Line plot for QBER vs loss
├── req.txt                          # Requirements list
├── LICENSE
└── README.md                        # This file
```

---

## 🚀 Run Instructions

### 1. Clone Repository
```bash
git clone https://github.com/your-username/Hybrid-QKD-Chaotic-Encryption-System.git
cd Hybrid-QKD-Chaotic-Encryption-System
```

### 2. Install Dependencies
```bash
pip install -r req.txt
```

### 3. Execute Simulation
```bash
python main.py
```

### Output:
* `qber_vs_loss.png`: Channel loss vs QBER curve
* `attack_qber.png`: Simulated attack QBERs
* Console output: BER, Avalanche effect, Monobit test

---

## 📊 Output Metrics
| Metric           | Description                                   |
| ---------------- | --------------------------------------------- |
| **BER**          | Bit Error Rate (post-decryption)              |
| **Avalanche**    | Cipher sensitivity to single-bit input change |
| **Monobit Test** | Checks keystream balance (0.5 mean)           |
| **QBER Graphs**  | Show impact of attacks and losses             |

---

---

## 📚 References
* Bennett & Brassard (1984) — BB84 Quantum Cryptography
* Baptista (1998) — Chaos-based secure communication
* Cho et al. (2014) — Gated Recurrent Units (GRU)
* Gisin et al. (2002) — QKD Security Survey

---

## 👤 Author
**Pratham Desai**

---

## 📜 License
MIT License – see `LICENSE` file.

> ⚠️ For academic use only – not suitable for production-grade cryptographic security.
