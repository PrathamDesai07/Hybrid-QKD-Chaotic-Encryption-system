# 🔐 Hybrid-QKD-Chaotic-Encryption-System
A complete multi-phase simulation of a hybrid quantum-classical encryption pipeline. This project integrates Quantum Key Distribution (QKD) with chaotic maps, machine learning, GRU-based keystream generation, and classical cryptographic techniques.

---

## 🔐 1. BB84 Protocol (Quantum Key Distribution)
The **BB84 protocol** is a quantum-safe cryptographic scheme used to generate a shared secret key between two parties over an insecure quantum channel. It exploits the **no-cloning theorem** and the nature of **quantum measurements** to detect the presence of an eavesdropper.

We simulate:
* **Photon loss and depolarization**
* **Error correction** using **LDPC codes**
* **Privacy amplification** to distill the final key

## 🧠 2. Chaotic Encryption
Chaos theory allows generation of **deterministic yet unpredictable sequences**, ideal for secure encryption. We use:
* **Chaotic Maps** (e.g., Chen, Rossler) for generating keystreams
* **Logistic Maps** for virtual noise masking
* **Feedback-Chaos Systems** for adaptive encryption

These systems mimic randomness and are sensitive to initial conditions, making reverse-engineering extremely difficult.

## 🤖 3. Machine Learning-based Map Selection (Phase‑6A)
We introduce **ML classification** to select the optimal chaotic map based on estimated **Quantum Bit Error Rate (QBER)**. This allows dynamic adaptation of encryption strategy based on channel conditions, increasing robustness.

## 🔁 4. GRU-based Keystream Generator (Phase‑6B)
A **GRU (Gated Recurrent Unit)** neural network is trained on chaotic sequences to **learn and replicate** non-linear chaotic behavior. This adds a layer of **AI-driven unpredictability** in keystream generation, further enhancing security.

## 🎲 5. Virtual Noise Layer (Phase‑6C)
A **logistic map** is used to generate a byte-level noise mask, which is added to the plaintext before encryption and removed after decryption. This introduces non-linearity and deters chosen-plaintext attacks.

## 🔄 6. Classical Cipher (Phase‑7)
After keystream generation:
* The plaintext is **scrambled**
* Masked using **virtual noise**
* Whitened using SHA-256
* Encrypted using a **2-round XOR cipher**
* Authenticated via **HMAC-SHA256**

We also simulate **attack models** like Intercept-Resend and Photon Splitting, plotting their impact on QBER.

## 🧪 System Goals
* Provide **quantum-resilient encryption**
* Ensure high **avalanche effect** and **bit-level unpredictability**
* Dynamically adapt to changing channel conditions via ML
* Maintain **low BER**, secure HMAC authentication, and **high randomness** in keystream

---

## 📌 Overview
This project combines **Quantum Key Distribution (QKD)** principles with **chaotic encryption** and **machine learning-based dynamic keystream adaptation** to create a **hybrid cryptographic system** that is resilient to both classical and quantum attacks.

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

## 📚 References
* Bennett & Brassard (1984) — BB84 Quantum Cryptography
* Baptista (1998) — Chaos-based secure communication
* Cho et al. (2014) — Gated Recurrent Units (GRU)
* Gisin et al. (2002) — QKD Security Survey

---

## 👤 Author
**Pratham Desai**
📧 [pratham@example.com](mailto:prathamdesai071204@gmail.com)
🔗 [LinkedIn](https://linkedin.com/in/your-profile)
🔗 [GitHub](https://github.com/PrathamDesai07)

---

## 📜 License
MIT License – see `LICENSE` file.
