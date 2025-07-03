```markdown
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
├── bb84\_pipeline.py                # BB84 QKD core logic
├── classical\_encryption\_engine.py  # Custom symmetric cipher engine
├── feedback\_chaos.py               # Feedback-chaotic stream generator
├── gru\_keystream.py                # GRU-based keystream
├── ml\_map\_selection.py             # ML-based chaotic map selector
├── virtual\_noise\_layer.py          # Noise masking & unmasking
├── final\_key.npy                   # Shared secret key
├── gru\_chaotic\_model.pth           # Saved PyTorch GRU model
├── attack\_qber.png                 # Bar graph of attack QBERs
├── qber\_vs\_loss.png                # Line plot for QBER vs loss
├── req.txt                         # Requirements list
├── LICENSE
└── README.md                       # This file

````

---

## 🚀 Run Instructions

### 1. Clone Repository
```bash
git clone https://github.com/your-username/Hybrid-QKD-Chaotic-Encryption-System.git
cd Hybrid-QKD-Chaotic-Encryption-System
````

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

## 📒 Phase 8: Report Deliverables

* ✅ Technical documentation
* ✅ System diagram
* ✅ Plots and metrics
* ✅ Supervisor Summary (one-pager)
* ✅ Full code with comments

📁 Suggested folders for report:

```
/qkd/        → BB84 protocol implementation  
/chaos/      → Chaotic maps, parameters  
/ml/         → ML models and training utils  
/analysis/   → Result graphs and evaluation code  
```

---

## 📚 References

* Bennett & Brassard (1984) — BB84 Quantum Cryptography
* Baptista (1998) — Chaos-based secure communication
* Cho et al. (2014) — Gated Recurrent Units (GRU)
* Gisin et al. (2002) — QKD Security Survey

---

## 👤 Author

**Pratham Desai**
📧 [pratham@example.com](mailto:pratham@example.com)
🔗 [LinkedIn](https://linkedin.com/in/your-profile)
🔗 [GitHub](https://github.com/your-username)

---

## 📜 License

MIT License – see `LICENSE` file.

> ⚠️ For academic use only – not suitable for production-grade cryptographic security.

```

---

Let me know if you’d like to include a `.pdf` for Phase 8 or an auto-generated summary script.
```
