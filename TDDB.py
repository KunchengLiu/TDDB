import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_tddb_monte_carlo(voltages, n_devices_per_v=500, t_max=1e5):
    """
    使用蒙特卡洛方法生成带有物理约束的 TDDB 数据集
    """
    A = 1e12       # E-model 的前置常数 (Scale factor)
    gamma = 4.5    # 电压加速因子
    beta = 1.3     # Weibull 形状参数 
    v_noise = 0.05 # 蒙特卡洛噪声：模拟局部工艺波动
    
    all_data = []

    for V in voltages:
        V_local = np.random.normal(loc=V, scale=v_noise, size=n_devices_per_v)
        eta_local = A * np.exp(-gamma * V_local)
        
        t_bd_ideal = eta_local * np.random.weibull(beta, n_devices_per_v)
        
        status = np.ones(n_devices_per_v)
        censored_mask = t_bd_ideal > t_max
        status[censored_mask] = 0        
        
        t_bd_observed = np.copy(t_bd_ideal)
        t_bd_observed[censored_mask] = t_max
        
        df_v = pd.DataFrame({
            'Stress_Voltage': V,
            'Actual_Voltage_Noise': V_local, 
            't_BD': t_bd_observed,
            'Status': status                
        })
        all_data.append(df_v)

    return pd.concat(all_data, ignore_index=True)

# ==========================================
# 1. 运行生成器
# ==========================================
stress_conditions = [4.5, 4.3, 4.1, 3.9, 3.0] 
df_tddb = generate_tddb_monte_carlo(voltages=stress_conditions, n_devices_per_v=200, t_max=1e8)

print("数据统计概览:")
print(df_tddb.groupby('Stress_Voltage')['Status'].value_counts())

# ==========================================
# 2. 可视化：Weibull 图 与 TF=eta 外推图
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

df_failed = df_tddb[df_tddb['Status'] == 1]

extracted_etas = []
fit_voltages = []
eta_verification = None

# ---------------- 图 1：Weibull 分布 ----------------
for v in stress_conditions:
    data_v = df_failed[df_failed['Stress_Voltage'] == v]['t_BD'].sort_values()
    
    if len(data_v) > 3:
        F = (np.arange(1, len(data_v) + 1) - 0.3) / (len(data_v) + 0.4) 
        
        W = np.log(-np.log(1 - F))
        ln_t = np.log(data_v)
        
        p = ax1.plot(ln_t, W, marker='o', linestyle='', label=f'{v}V Data')
        color = p[0].get_color() 
        
        beta_fit, C_fit = np.polyfit(ln_t, W, 1)
        ax1.plot(ln_t, beta_fit * ln_t + C_fit, linestyle='--', color=color, alpha=0.6)
        
        eta_fit = np.exp(-C_fit / beta_fit)
        
        if v > 3.0:
            extracted_etas.append(eta_fit)
            fit_voltages.append(v)
        else:
            eta_verification = eta_fit

ax1.set_xlabel('ln(t_BD)')
ax1.set_ylabel('ln(-ln(1-F(t)))')
ax1.set_title('Weibull Distribution & Linear Fit')
ax1.legend()
ax1.grid(True)

# ---------------- 图 2：TF = eta 外推图 (E-model) ----------------
if len(fit_voltages) >= 2:
    ln_etas = np.log(extracted_etas)
    neg_gamma, ln_A = np.polyfit(fit_voltages, ln_etas, 1)
    gamma_fit = -neg_gamma
    
    v_extrap = np.linspace(2.8, 4.6, 100)
    ln_eta_extrap = ln_A - gamma_fit * v_extrap
    
    # 【重点修复区：加上了 fr 和 r 前缀】
    ax2.plot(v_extrap, ln_eta_extrap, 'b--', label=fr'E-Model Fit ($\gamma \approx {gamma_fit:.2f}$)')
    ax2.plot(fit_voltages, ln_etas, 'ro', markersize=8, label=r'Extracted $\eta$ (High V)')
    
    if eta_verification is not None:
        ax2.plot(3.0, np.log(eta_verification), 'g*', markersize=12, label=r'Actual $\eta$ @ 3.0V')

ax2.set_xlabel('Stress Voltage (V)')
# 【重点修复区：加上了 r 前缀】
ax2.set_ylabel(r'ln($\eta$) [Time to 63.2% Failure]')
ax2.set_title('Voltage Acceleration (E-model) Extrapolation')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()