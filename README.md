## KCC 2025 GConvLSTM with Collapsed Coreness Problem

```
# GComvLSTM_cc
python3 main.py --network karate --method GConvLSTM_partial --sample_rate 0.1 --gpu 0 --hop 1&
```
```
# GCN_cc
python3 main.py --network karate --method GCN_sample --sample_rate 1.0 --gpu 0 --hop 0&
```
- GCC: F.Zhang et al., "Discovering key users for defending network structural stability.", WWW 2022.
