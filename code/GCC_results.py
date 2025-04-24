import networkx as nx
import argparse
import copy
import pandas as pd
import csv
import func

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='karate', help='network file path')
    args = parser.parse_args()
    for network in ["strike","mexican","karate","football","polblogs","wisconsin",'facebook']:
    # for network in ['facebook']:
        file_name = f'/home/hwhwkim7/KCC_2025/output/GCC/{network}_11.txt'
        G = func.load_graph(f'../dataset/{network}/network.dat')
        K = nx.core_number(G)

        results = [0]
        id_list = [0]
        follower_list = [0]
        follower_computation_list = []
        times = 0
        with open(file_name, 'r') as f:
            for line in f:
                parts = line.strip().split("\t")
                id_value = int(parts[2])  # 세 번째 항목이 id
                count = int(parts[3])  # 네 번째 항목이 follower computations
                id_list.append(id_value)
                follower_computation_list.append(count)
                G_ = copy.deepcopy(G)
                G_.remove_nodes_from(id_list)
                K_ = nx.core_number(G_)
                follower = sum(K[n] - K_[n] for n in K_)
                follower_list.append(follower)
                temp_times = float(parts[3])
                results.append([int(parts[0]), follower, round(temp_times-times,5), temp_times])
                times = temp_times
        print(id_list)
        print(follower_list)
        # df = pd.DataFrame(results)
        # df.insert(0, "dataset", args.network)
        # df.insert(1, "method", 'GCC')
        #
        # output_file = '../output/GCC_results.csv'
        # df.to_csv(output_file, index=False, header=False, mode='a')
        # print(f'Save results to {output_file}')
        with open('../output/GCC.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            for i in range(len(follower_computation_list)):
                row = [network, "GCC", 1.0, i, follower_list[i], follower_computation_list[i]]
                writer.writerow(row)

if __name__ == '__main__':
    main()





