# My NETCONF and SNMP research sample:
import numpy as np
from scipy.optimize import minimize

# パラメータ
T = 100  # 時間単位の数
N_SNMP = 50  # SNMPデバイスの数
N_NETCONF = 50  # NETCONFデバイスの数

# 確率変数
def generate_random_requests():
    return np.random.poisson(lam=5, size=(T, N_SNMP + N_NETCONF))

# コスト関数
def cost_function(x, requests):
    snmp_allocation = x[:T]
    netconf_allocation = x[T:]
    
    snmp_cost = np.sum(np.maximum(0, requests[:, :N_SNMP].sum(axis=1) - snmp_allocation))
    netconf_cost = np.sum(np.maximum(0, requests[:, N_SNMP:].sum(axis=1) - netconf_allocation))
    
    return snmp_cost + netconf_cost

# 制約条件
def constraint(x):
    return np.sum(x) - (T * (N_SNMP + N_NETCONF))  # 総リソース制約

# 最適化問題
def optimize_resource_allocation(num_scenarios):
    scenarios = [generate_random_requests() for _ in range(num_scenarios)]
    
    def objective(x):
        return np.mean([cost_function(x, scenario) for scenario in scenarios])
    
    x0 = np.ones(2*T) * (N_SNMP + N_NETCONF) / 2  # 初期解
    cons = {'type': 'eq', 'fun': constraint}
    bounds = [(0, None) for _ in range(2*T)]
    
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
    
    return result.x

# モデルの実行
num_scenarios = 100
optimal_allocation = optimize_resource_allocation(num_scenarios)

# 結果の解釈
snmp_allocation = optimal_allocation[:T]
netconf_allocation = optimal_allocation[T:]

print(f"Average SNMP resource allocation: {np.mean(snmp_allocation):.2f}")
print(f"Average NETCONF resource allocation: {np.mean(netconf_allocation):.2f}")