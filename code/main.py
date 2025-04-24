import argparse
import numpy as np
import torch
import torch.nn as nn
import csv

import func

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='karate', help='network file path')
    parser.add_argument('--method', default='GConvLSTM_combine')
    parser.add_argument('--budget', default=10, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hop', default=1, type=int)
    parser.add_argument('--lambda_ewc', default=400, type=int)
    parser.add_argument('--initial_epochs', default=200, type=int)
    parser.add_argument('--continual_epochs', default=100, type=int)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--use_ewc', default=True, type=bool)
    parser.add_argument('--initial_sample_rate', default=0.1, type=float)
    parser.add_argument('--sample_rate', default=0.1, type=float)
    # parser.add_argument('--continual_sample_rate', default=0.5, type=float)


    args = parser.parse_args()

    test_graph = func.load_graph(f"../dataset/{args.network}/network.dat")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    budget = args.budget
    criterion = nn.MSELoss()
    hop = args.hop
    lambda_ewc = args.lambda_ewc
    initial_epochs = args.initial_epochs
    continual_epochs = args.continual_epochs
    learning_rate = args.learning_rate
    use_ewc = args.use_ewc
    sample_rate = args.sample_rate
    initial_sample_rate = args.sample_rate

    coreness_loss_list = []
    follower_computation_list = []

    for iteration in range(10):
        # print(iteration)
        if args.method == 'GConvLSTM_partial':
            # follower 기반 hop으로 subgraph 생성
            # print("GConvLSTM Partial")
            coreness_loss, removed_nodes, _, follower_computations = func.GConvLSTM_partial(device, test_graph.copy(), budget, "GConvLSTM",None, criterion,
                                                                             initial_sample_rate, sample_rate, hop, lambda_ewc, initial_epochs, continual_epochs, learning_rate, use_ewc)
        elif args.method == 'GConvLSTM_combine':
            # print("GConvLSTM Combined")
            coreness_loss, removed_nodes, _, follower_computations = func.GConvLSTM_combine(device, test_graph.copy(), budget, "GConvLSTM",None, criterion,
                                                                                initial_sample_rate, sample_rate, hop, lambda_ewc, initial_epochs, continual_epochs, learning_rate, use_ewc)
        elif args.method == 'GCN_sample':
            # print("GCN Sample")
            coreness_loss, removed_nodes, _, follower_computations = func.GCN_sample(device, test_graph.copy(), budget, "GCN",None, criterion,
                                                                     initial_sample_rate, sample_rate, hop, lambda_ewc, initial_epochs, continual_epochs, learning_rate, use_ewc)
        coreness_loss_list.append(coreness_loss)
        follower_computation_list.append(follower_computations)

    coreness_loss_list_mean = np.mean(coreness_loss_list, axis=0)
    follower_computation_list_mean = np.mean(follower_computation_list, axis=0)
    print(coreness_loss_list_mean, follower_computation_list_mean)

    with open(f'../output/result_final_hop.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(coreness_loss_list_mean)):
            row = [args.network, args.method, args.sample_rate, args.hop, i, coreness_loss_list_mean[i], follower_computation_list_mean[i]]
            writer.writerow(row)

if __name__ == '__main__':
    main()