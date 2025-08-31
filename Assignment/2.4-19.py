import numpy as np
import networkx as nx

N = 4

G = nx.grid_2d_graph(N, N)

nodes = sorted(G.nodes())
node_map = {node: i for i, node in enumerate(nodes)}

L = nx.laplacian_matrix(G).toarray()

ground_node_idx = node_map[(1, 1)]   
current_node_idx_orig = node_map[(2, 2)]

print(f"格栅大小: {N}x{N}")
print(f"接地节点 (2,2) 的原始索引: {ground_node_idx}")
print(f"电流注入节点 (3,3) 的原始索引: {current_node_idx_orig}")
print("-" * 20)
K = np.delete(L, ground_node_idx, axis=0)
K = np.delete(K, ground_node_idx, axis=1)

num_nodes_remaining = N * N - 1
f = np.zeros(num_nodes_remaining)

if current_node_idx_orig > ground_node_idx:
    current_node_idx_new = current_node_idx_orig - 1
else:
    current_node_idx_new = current_node_idx_orig
    
print(K)
f[current_node_idx_new] = 1

print(f"缩减后的矩阵 K 的维度: {K.shape}")
print(f"电流注入节点 (3,3) 的新索引: {current_node_idx_new}")
print("-" * 20)

u = np.linalg.solve(K, f)

voltage_at_3_3 = u[current_node_idx_new]

print(f"求解得到的电压 u(3,3) = {voltage_at_3_3:.4f}")
print(f"因此，(2,2) 和 (3,3) 节点间的等效电阻是: {(voltage_at_3_3 - 0) / 1:.4f} ohms (假设单位电阻为1 ohm)")