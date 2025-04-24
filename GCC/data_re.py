import networkx as nx
import argparse

def load_graph(filename):
    """karate.dat 파일을 로드하여 NetworkX 그래프로 변환"""
    G = nx.read_edgelist(filename, nodetype=int)

    # 셀프 루프 제거
    G.remove_edges_from(nx.selfloop_edges(G))

    # 노드 라벨 정리 (예: 0부터 시작하는 연속된 정수로 변경)
    mapping = {node: idx for idx, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    return G

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='facebook', help='network file path')
    args = parser.parse_args()

    G = load_graph(f"../dataset/{args.network}/network.dat")


    with open(f'../dataset/{args.network}/network_re.dat', mode='w', newline='') as file:
        file.write(f'{nx.number_of_nodes(G)} {nx.number_of_edges(G)}\n')
        for u, v in G.edges():
            file.write(f'{u} {v}\n')