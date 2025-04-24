gpu = 0
with open('cmds.txt', 'w') as f:
    for network in ['karate', 'football', 'mexican', 'strike']:
    # for network in ['wisconsin', 'polblogs']:
    # for network in ['facebook']:
        for hop in [1, 2]:
            for sample_rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
                for method in ['GConvLSTM_partial']:
                    f.write(f'python3 main.py --network {network} --method {method} --sample_rate {sample_rate} --gpu {gpu%4} --hop {hop}&\n')
                    gpu += 1

    for network in ['karate', 'football', 'mexican', 'strike']:
    # for network in ['wisconsin', 'polblogs']:
    # for network in ['facebook']:
        for method in ['GCN_sample']:
            f.write(f'python3 main.py --network {network} --method {method} --sample_rate {1.0} --gpu {gpu%4} --hop {0}&\n')
            gpu += 1