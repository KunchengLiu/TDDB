import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_tddb_monte_carlo(voltages, n_devices_per_v=500, t_max=1e5):
    """
    使用蒙特卡洛方法生成带有物理约束的 TDDB 数据集
    
    参数:
    voltages: list, 测试电压列表 (例如 [4.5, 4.3, 4.1, 3.5])
    n_devices_per_v: int, 每个电压下测试的器件数量
    t_max: float, 最大测试时间 (秒)，用于生成右侧截尾数据
    """
    
    # --- 物理与工艺参数设定 (你可以根据文献微调这些值) ---
    A = 1e12       # E-model 的前置常数 (Scale factor)
    gamma = 4.5    # 电压加速因子 (决定了降压时寿命延长的倍数)
    beta = 1.3     # Weibull 形状参数 (通常在 1.1 ~ 1.5 之间，代表磨损失效)
    v_noise = 0.05 # 蒙特卡洛噪声：模拟局部工艺波动 (比如氧化层厚度不均导致的有效电压波动)
    
    all_data = []

    for V in voltages:
        # 1. 蒙特卡洛注入：生成带有局部工艺变异的实际承受电压
        # 使用高斯分布模拟器件间的差异
        V_local = np.random.normal(loc=V, scale=v_noise, size=n_devices_per_v)
        
        # 2. 物理模型约束：计算每个器件的特征寿命 eta (Scale parameter)
        eta_local = A * np.exp(-gamma * V_local)
        
        # 3. Weibull 随机抽样：生成理想的击穿时间 t_bd
        # numpy 的 weibull 函数生成的是标准 weibull，需要乘以 eta
        t_bd_ideal = eta_local * np.random.weibull(beta, n_devices_per_v)
        
        # 4. 引入右侧截尾 (Right-Censoring)
        status = np.ones(n_devices_per_v) # 1 代表发生了击穿 (Failed)
        
        # 找出那些超过最大测试时间的器件
        censored_mask = t_bd_ideal > t_max
        status[censored_mask] = 0         # 0 代表未击穿 (Censored)
        
        # 将被截尾的器件的时间强制设为最大测试时间
        t_bd_observed = np.copy(t_bd_ideal)
        t_bd_observed[censored_mask] = t_max
        
        # 记录数据
        df_v = pd.DataFrame({
            'Stress_Voltage': V,
            'Actual_Voltage_Noise': V_local, # 真实世界的隐藏变量，你的神经网络不知道这个值
            't_BD': t_bd_observed,
            'Status': status                 # 用于你的 Survival Analysis Loss Function
        })
        all_data.append(df_v)

    return pd.concat(all_data, ignore_index=True)

# ==========================================
# 运行生成器
# ==========================================
# 假设我们测 4 个高压用于训练，1 个低压用于外推验证
stress_conditions = [4.5, 4.3, 4.1, 3.9, 3.0] 
df_tddb = generate_tddb_monte_carlo(voltages=stress_conditions, n_devices_per_v=200, t_max=10000)

print(df_tddb.head())
print("\n数据统计概览:")
print(df_tddb.groupby('Stress_Voltage')['Status'].value_counts())

# ==========================================
# 简单可视化 (Weibull Plot 的雏形)
# ==========================================
# 过滤掉 Censored 数据用于简单展示
df_failed = df_tddb[df_tddb['Status'] == 1]
for v in stress_conditions:
    data_v = df_failed[df_failed['Stress_Voltage'] == v]['t_BD'].sort_values()
    if len(data_v) > 0:
        # 计算累积失效概率 F(t)
        F = (np.arange(1, len(data_v) + 1) - 0.3) / (len(data_v) + 0.4) 
        # Weibull 变换: W = ln(-ln(1-F))
        W = np.log(-np.log(1 - F))
        plt.plot(np.log(data_v), W, marker='o', linestyle='', label=f'{v}V')

plt.xlabel('ln(t_BD)')
plt.ylabel('ln(-ln(1-F(t)))')
plt.title('Synthetic TDDB Weibull Distribution')
plt.legend()
plt.grid(True)
plt.show()