# LSND - Liner Shipping Network Design

Liner Shipping Network Design Problem (LSNDP) 求解框架，基于 Brouer et al. (2014) 论文的模型复现与实现。

## 研究背景

班轮航运网络设计问题 (LSNDP) 是为一组集装箱船队创建循环航线，使得运输收入最大化、运营成本最小化。该问题已被证明为强 NP-hard。

**核心论文:**
> Brouer, B. D., Alvarez, J. F., Plum, C. E. M., Pisinger, D., & Sigurd, M. M. (2014). *A Base Integer Programming Model and Benchmark Suite for Liner-Shipping Network Design.* Transportation Science, 48(2), 281-312.

## 项目结构

```
LSND/
├── data/LinerLib/           # LINER-LIB 基准数据集 (7个实例)
├── src/
│   ├── data_reader.py       # 数据加载模块
│   ├── model/
│   │   ├── mip_model.py     # MIP 主模型 (论文 §6.1, 公式 2-10)
│   │   ├── mcfp.py          # 多商品流问题求解
│   │   └── route_generation.py  # 路径生成 AUX 模型 (论文 §6.3, 公式 11-38)
│   ├── algorithm/
│   │   ├── metaheuristic.py # 禁忌搜索 + 启发式列生成 (论文 §6.2)
│   │   ├── tabu_search.py   # 禁忌搜索组件
│   │   └── solver.py        # 统一求解器接口
│   └── utils/
│       ├── cost_calculator.py   # 成本计算 (燃油、港口、运河等)
│       └── network_builder.py   # MCF 网络图构建
├── results/                 # 求解结果输出
├── tests/                   # 测试
├── app.py                   # Streamlit 交互式应用
├── requirements.txt         # Python 依赖
└── LinerLib/                # 原始 LINER-LIB (含 C++ 参考实现)
```

## 模型架构

### MIP 主模型 (公式 2-10)

**目标函数** (最小化):
- (2a) 船舶 TC 费用 + 未使用船舶闲置/租出收入
- (2b) 燃油(航行+停泊) + 运河费 + 靠港费
- (2c) 拒运惩罚 - 运输收入
- (2d) 装卸费 + 转运费

**约束**:
- (3) 流量守恒 (支持蝴蝶航线的三索引流量)
- (4) 到达目的地流量
- (5) 需求满足 (运输 + 拒绝 = 总需求)
- (6) 边容量约束
- (7) 船舶数量约束

### 路径生成 AUX 模型 (公式 11-38)

为每种船型、速度和船舶数组合，生成最优化的循环航线：
- 支持蝴蝶航线 (最多一个蝴蝶节点)
- MTZ 子环消除约束
- 周频/双周频约束 (大船 >=1200 FFE 必须周频)
- FFE-miles 容量平衡

### 元启发式算法 (§6.2)

1. 启发式列生成: 通过 AUX 模型批量生成候选航线
2. MCFP 评估: 对每条候选航线求解多商品流问题
3. 禁忌搜索: 避免循环，支持回溯到最优解

### 成本公式

- **燃油**: `F(s) = (s/v*)^3 * f*` (立方函数), 价格 600 USD/ton
- **TC 费用**: TC_rate x 规划期(180天)
- **港口费**: 固定费 + 变动费 x 容量
- **拒运惩罚**: 1000 USD/FFE

## 基准实例

| 实例 | 港口 | OD对 | 需求 (FFE/周) | 运力 (FFE) | 供需比 |
|------|------|------|-------------|-----------|--------|
| Baltic | 12 | 22 | 4,904 | 3,400 | 0.69 |
| WAF | 20 | 37 | 8,541 | 28,700 | 3.36 |
| Mediterranean | 39 | 365 | 7,545 | 14,800 | 1.96 |
| Pacific | 45 | 722 | 44,180 | 151,800 | 3.44 |
| EuropeAsia | 114 | 4,000 | 76,944 | 425,900 | 5.54 |
| WorldSmall | 47 | 1,764 | 128,281 | 611,800 | 4.77 |
| WorldLarge | 201 | 9,622 | 138,914 | 1,071,100 | 7.71 |

## 安装与运行

```bash
# 安装依赖
pip install -r requirements.txt
brew install cbc  # macOS, 安装 CBC 求解器

# 运行测试
python -m tests.test_data

# 命令行求解
python -c "
from src.algorithm.solver import solve_instance
result = solve_instance('Baltic', method='metaheuristic', max_time=120)
print(f'Objective: {result.objective:.0f}')
print(f'Rotations: {len(result.rotations)}')
"

# 启动可视化应用
streamlit run app.py
```

## 求解结果输出

求解结果包含与论文 Table 9 对应的成本分解:

| 符号 | 含义 |
|------|------|
| Z | 总目标值 |
| Q | 总收入 |
| c_v | 船舶 TC 费用 |
| c_b | 燃油费用 |
| c_p | 港口费用 |
| c_c | 运河费用 |
| c_m | 装卸费用 |
| c_t | 转运费用 |
| L_v | 未使用船舶价值 |
| F | 被拒绝运量 (FFE) |

## 依赖

- Python >= 3.10
- pandas >= 2.0
- plotly >= 5.0
- streamlit >= 1.30
- pulp >= 2.7
- CBC solver (via Homebrew or system package)
