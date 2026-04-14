"""
LINERLIB 班轮航运网络设计应用
使用 Streamlit + Plotly 实现交互式可视化与求解。
启动: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
import time
from pathlib import Path

from src.data_reader import load_instance, load_distances, INSTANCES

# 论文 Table 9 参考值 — Low / Base / High 三种场景
# 单位: USD (论文中为 k$, 此处 ×1000)
PAPER_BENCHMARKS = {
    "Baltic": {
        "Low":  {"Z_best": -6_044_000,  "Z_median": 838_000,    "Q_best": 96_750_000,  "rotations_best": 5, "reject_pct_best": 8.6},
        "Base": {"Z_best": -8_365_000,  "Z_median": -6_582_000, "Q_best": 98_310_000,  "rotations_best": 3, "reject_pct_best": 5.0, "source": "Baltic_best_base.log"},
        "High": {"Z_best": -15_678_000, "Z_median": -11_509_000,"Q_best": 103_500_000, "rotations_best": 5, "reject_pct_best": 0.7},
    },
    "WAF": {
        "Low":  {"Z_best": -115_361_000, "Z_median": -109_470_000, "Q_best": 345_800_000, "rotations_best": 7,  "reject_pct_best": 11.7},
        "Base": {"Z_best": -143_110_000, "Z_median": -137_681_000, "Q_best": 370_600_000, "rotations_best": 11, "reject_pct_best": 4.7},
        "High": {"Z_best": -159_944_000, "Z_median": -152_635_000, "Q_best": 382_000_000, "rotations_best": 11, "reject_pct_best": 1.6},
    },
    "Mediterranean": {
        "Low":  {"Z_best": 29_504_000,  "Z_median": 42_636_000,  "Q_best": 128_900_000, "rotations_best": 7, "reject_pct_best": 7.4},
        "Base": {"Z_best": 12_209_000,  "Z_median": 24_934_000,  "Q_best": 136_800_000, "rotations_best": 7, "reject_pct_best": 1.2},
        "High": {"Z_best": 6_606_000,   "Z_median": 14_267_000,  "Q_best": 137_100_000, "rotations_best": 11,"reject_pct_best": 0.9},
    },
    "Pacific": {
        "Low":  {"Z_best": 84_615_000,   "Z_median": 132_020_000,  "Q_best": 1_120_000_000, "rotations_best": 22, "reject_pct_best": 7.8},
        "Base": {"Z_best": -54_087_000,  "Z_median": -12_774_000,  "Q_best": 1_197_000_000, "rotations_best": 21, "reject_pct_best": 3.1},
        "High": {"Z_best": -101_671_000, "Z_median": -75_321_000,  "Q_best": 1_208_000_000, "rotations_best": 20, "reject_pct_best": 2.0},
    },
    "WorldSmall": {
        "Low":  {"Z_best": -551_842_000,  "Z_median": -417_100_000,  "Q_best": 4_917_000_000, "rotations_best": 35, "reject_pct_best": 19.6},
        "Base": {"Z_best": -888_669_000,  "Z_median": -888_669_000,  "Q_best": 5_199_000_000, "rotations_best": 40, "reject_pct_best": 14.8},
        "High": {"Z_best": -1_168_067_000,"Z_median": -1_168_067_000,"Q_best": 5_266_000_000, "rotations_best": 37, "reject_pct_best": 13.1},
    },
    "EuropeAsia": {
        "Low":  {"Z_best": -361_041_000,  "Z_median": -228_186_000,  "Q_best": 3_109_000_000, "rotations_best": 30, "reject_pct_best": 14.5},
        "Base": {"Z_best": -657_972_000,  "Z_median": -561_417_000,  "Q_best": 3_283_000_000, "rotations_best": 35, "reject_pct_best": 10.4},
        "High": {"Z_best": -766_385_000,  "Z_median": -670_099_000,  "Q_best": 3_390_000_000, "rotations_best": 37, "reject_pct_best": 7.6},
    },
}

st.set_page_config(page_title="LINERLIB 班轮航运网络设计", layout="wide")
st.title("LINERLIB 班轮航运网络设计")

# --- 侧边栏：选择实例与参数 ---
st.sidebar.header("数据集选择")
instance_name = st.sidebar.selectbox("选择基准实例", INSTANCES, index=0)

data = load_instance(instance_name)
ports = data["ports"]
ports_all = data["ports_all"]
demand = data["demand"]
fleet = data["fleet"]

st.sidebar.markdown(f"**港口数**: {len(ports)}  \n**OD 需求对**: {len(demand)}  \n**船型数**: {len(fleet)}")

# --- 侧边栏：可视化选项 ---
st.sidebar.header("可视化选项")
show_demand_links = st.sidebar.checkbox("显示需求航线", value=True)

demand_min = int(demand["FFEPerWeek"].min())
demand_max = int(demand["FFEPerWeek"].max())
if demand_min < demand_max:
    ffe_range = st.sidebar.slider(
        "FFE/周 筛选范围", demand_min, demand_max, (demand_min, demand_max)
    )
else:
    ffe_range = (demand_min, demand_max)

top_n_routes = st.sidebar.slider("显示 Top N 航线", 5, min(200, len(demand)), min(50, len(demand)))

# 筛选需求
demand_filtered = demand[
    (demand["FFEPerWeek"] >= ffe_range[0]) & (demand["FFEPerWeek"] <= ffe_range[1])
].nlargest(top_n_routes, "FFEPerWeek")

# --- Tab 布局 ---
tab_map, tab_demand, tab_fleet, tab_ports, tab_solver, tab_results = st.tabs(
    ["网络地图", "需求分析", "船队信息", "港口详情", "求解器", "求解结果"]
)

# ========== Tab 1: 网络地图 ==========
with tab_map:
    port_coords = ports_all.set_index("UNLocode")[["Longitude", "Latitude", "name"]].to_dict("index")

    fig = go.Figure()

    # 绘制求解结果航线 (如果有)
    if "solve_result" in st.session_state and st.session_state.solve_result is not None:
        show_solution = st.checkbox("叠加显示求解结果航线", value=True)
        if show_solution:
            result = st.session_state.solve_result
            colors = px.colors.qualitative.Set2
            for idx, rot_detail in enumerate(result.rotation_details):
                pc = rot_detail["port_calls"]
                color = colors[idx % len(colors)]
                # 绘制航线边
                for k in range(len(pc)):
                    i_p = pc[k]
                    j_p = pc[(k + 1) % len(pc)]
                    if i_p in port_coords and j_p in port_coords:
                        ic, jc = port_coords[i_p], port_coords[j_p]
                        fig.add_trace(go.Scattergeo(
                            lon=[ic["Longitude"], jc["Longitude"]],
                            lat=[ic["Latitude"], jc["Latitude"]],
                            mode="lines",
                            line=dict(width=3, color=color),
                            hoverinfo="text",
                            text=f"航线 {rot_detail['id']}<br>{ic['name']} -> {jc['name']}<br>"
                                 f"船型: {rot_detail['vessel_class']}, {rot_detail['speed']}kn",
                            showlegend=False,
                        ))
                # 航线港口标记
                rot_lons = [port_coords[p]["Longitude"] for p in pc if p in port_coords]
                rot_lats = [port_coords[p]["Latitude"] for p in pc if p in port_coords]
                rot_names = [port_coords[p]["name"] for p in pc if p in port_coords]
                fig.add_trace(go.Scattergeo(
                    lon=rot_lons, lat=rot_lats,
                    mode="markers",
                    marker=dict(size=12, color=color, symbol="diamond",
                                line=dict(width=1.5, color="white")),
                    hovertext=rot_names,
                    hoverinfo="text",
                    name=f"{rot_detail['vessel_class']} ({rot_detail['speed']}kn)",
                ))

    # 绘制需求航线
    if show_demand_links:
        max_ffe = demand_filtered["FFEPerWeek"].max() if len(demand_filtered) > 0 else 1
        for _, row in demand_filtered.iterrows():
            o, d = row["Origin"], row["Destination"]
            if o in port_coords and d in port_coords:
                oc, dc = port_coords[o], port_coords[d]
                width = max(0.5, 4 * row["FFEPerWeek"] / max_ffe)
                opacity = max(0.2, 0.8 * row["FFEPerWeek"] / max_ffe)
                fig.add_trace(go.Scattergeo(
                    lon=[oc["Longitude"], dc["Longitude"]],
                    lat=[oc["Latitude"], dc["Latitude"]],
                    mode="lines",
                    line=dict(width=width, color=f"rgba(31,119,180,{opacity})"),
                    hoverinfo="text",
                    text=f"{oc['name']} -> {dc['name']}<br>FFE/周: {row['FFEPerWeek']}<br>收入: {row['Revenue_1']}",
                    showlegend=False,
                ))

    # 绘制港口点
    fig.add_trace(go.Scattergeo(
        lon=ports["Longitude"],
        lat=ports["Latitude"],
        mode="markers+text",
        marker=dict(size=7, color="crimson", line=dict(width=0.5, color="white")),
        text=ports["UNLocode"],
        textposition="top center",
        textfont=dict(size=8),
        hoverinfo="text",
        hovertext=ports.apply(
            lambda r: f"{r['name']} ({r['UNLocode']})<br>国家: {r['Country']}<br>吃水: {r['Draft']}m", axis=1
        ),
        name="港口",
    ))

    fig.update_layout(
        geo=dict(
            showland=True, landcolor="rgb(243,243,243)",
            showocean=True, oceancolor="rgb(204,229,255)",
            showcountries=True, countrycolor="rgb(204,204,204)",
            showcoastlines=True, coastlinecolor="rgb(150,150,150)",
            projection_type="natural earth",
        ),
        height=650,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(x=0.01, y=0.99),
    )

    if len(ports) > 0:
        lon_center = ports["Longitude"].mean()
        lat_center = ports["Latitude"].mean()
        lon_range = ports["Longitude"].max() - ports["Longitude"].min()
        lat_range = ports["Latitude"].max() - ports["Latitude"].min()
        scale = max(lon_range, lat_range)
        if scale < 60:
            fig.update_geos(
                center=dict(lon=lon_center, lat=lat_center),
                projection_scale=max(1, 150 / scale),
            )

    st.plotly_chart(fig, width='stretch')
    st.caption(f"当前显示 {len(demand_filtered)} 条需求航线（按 FFE/周 排序 Top {top_n_routes}）")

# ========== Tab 2: 需求分析 ==========
with tab_demand:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("需求分布 (FFE/周)")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=demand["FFEPerWeek"], nbinsx=40, marker_color="steelblue"))
        fig_hist.update_layout(xaxis_title="FFE/周", yaxis_title="航线数", height=350)
        st.plotly_chart(fig_hist, width='stretch')

    with col2:
        st.subheader("收入 vs 运量")
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=demand["FFEPerWeek"], y=demand["Revenue_1"],
            mode="markers", marker=dict(size=5, opacity=0.6, color="teal"),
            hovertext=demand.apply(lambda r: f"{r['Origin']}->{r['Destination']}", axis=1),
            hoverinfo="text+x+y",
        ))
        fig_scatter.update_layout(xaxis_title="FFE/周", yaxis_title="单箱收入", height=350)
        st.plotly_chart(fig_scatter, width='stretch')

    st.subheader("港口吞吐量排名 (Top 20)")
    outbound = demand.groupby("Origin")["FFEPerWeek"].sum().rename("出口FFE/周")
    inbound = demand.groupby("Destination")["FFEPerWeek"].sum().rename("进口FFE/周")
    throughput = pd.concat([outbound, inbound], axis=1).fillna(0)
    throughput["总吞吐量"] = throughput["出口FFE/周"] + throughput["进口FFE/周"]
    throughput = throughput.sort_values("总吞吐量", ascending=False).head(20)

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=throughput.index, y=throughput["出口FFE/周"], name="出口", marker_color="steelblue"))
    fig_bar.add_trace(go.Bar(x=throughput.index, y=throughput["进口FFE/周"], name="进口", marker_color="coral"))
    fig_bar.update_layout(barmode="stack", height=400, xaxis_title="港口", yaxis_title="FFE/周")
    st.plotly_chart(fig_bar, width='stretch')

    st.subheader("需求数据明细")
    st.dataframe(demand.sort_values("FFEPerWeek", ascending=False), width='stretch', height=300)

# ========== Tab 3: 船队信息 ==========
with tab_fleet:
    st.subheader(f"{instance_name} 实例 -- 可用船队")
    st.dataframe(fleet, width='stretch')

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("船型容量与数量")
        fig_fleet = go.Figure()
        fig_fleet.add_trace(go.Bar(
            x=fleet["Vessel class"], y=fleet["Capacity FFE"],
            text=fleet["Quantity"].astype(str) + " 艘",
            textposition="outside",
            marker_color="darkorange",
        ))
        fig_fleet.update_layout(xaxis_title="船型", yaxis_title="容量 (FFE)", height=400)
        st.plotly_chart(fig_fleet, width='stretch')

    with col2:
        st.subheader("日租金 vs 容量")
        fig_tc = go.Figure()
        fig_tc.add_trace(go.Scatter(
            x=fleet["Capacity FFE"],
            y=fleet["TC rate daily (fixed Cost)"],
            mode="markers+text",
            text=fleet["Vessel class"],
            textposition="top center",
            marker=dict(size=fleet["Quantity"] * 3, color="purple", opacity=0.7),
        ))
        fig_tc.update_layout(xaxis_title="容量 (FFE)", yaxis_title="日租金 (USD)", height=400)
        st.plotly_chart(fig_tc, width='stretch')

    total_capacity = (fleet["Capacity FFE"] * fleet["Quantity"]).sum()
    total_vessels = fleet["Quantity"].sum()
    total_demand_ffe = demand["FFEPerWeek"].sum()

    # 如果有求解结果，显示部署对比
    if "solve_result" in st.session_state and st.session_state.solve_result is not None:
        result = st.session_state.solve_result
        st.subheader("船队部署情况")
        usage_data = []
        for v_name, info in result.fleet_usage.items():
            usage_data.append({
                "船型": v_name,
                "可用": info["available"],
                "已部署": info["deployed"],
                "闲置": info["available"] - info["deployed"],
            })
        st.dataframe(pd.DataFrame(usage_data), width='stretch')

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("总运力 (FFE)", f"{total_capacity:,.0f}",
                  delta=f"供需比: {total_capacity/total_demand_ffe:.2f}" if total_demand_ffe > 0 else None)
    with col_m2:
        st.metric("船舶总数", f"{total_vessels}")

# ========== Tab 4: 港口详情 ==========
with tab_ports:
    st.subheader("实例涉及港口")
    st.dataframe(ports.sort_values("name"), width='stretch', height=400)

    st.subheader("全部港口数据库")
    search = st.text_input("搜索港口 (名称或代码)")
    display_ports = ports_all
    if search:
        mask = (
            ports_all["name"].str.contains(search, case=False, na=False) |
            ports_all["UNLocode"].str.contains(search, case=False, na=False)
        )
        display_ports = ports_all[mask]
    st.dataframe(display_ports.sort_values("name"), width='stretch', height=400)
    st.caption(f"共 {len(display_ports)} 个港口")

# ========== Tab 5: 求解器 ==========
with tab_solver:
    st.subheader("LSNDP 求解器")
    st.markdown("""
    基于 Brouer et al. (2014) 论文模型的求解器。
    - **元启发式**: 禁忌搜索 + 启发式列生成 (适用于所有实例)
    - **MIP**: 先生成候选航线再用整数规划求解 (适用于小实例)
    """)

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        method = st.selectbox("求解方法", ["metaheuristic", "mip"],
                             format_func=lambda x: "元启发式 (Tabu Search + Column Generation)" if x == "metaheuristic" else "MIP (整数规划)")
        max_time = st.slider("最大运行时间 (秒)", 30, 600, 120, step=30)
        max_iterations = st.slider("最大迭代数", 10, 200, 50) if method == "metaheuristic" else 1

    with col_s2:
        fuel_price = st.number_input("燃油价格 (USD/ton)", value=600, step=50)
        planning_horizon = st.number_input("规划期 (天)", value=180, step=30)
        reject_penalty = st.number_input("拒运惩罚 (USD/FFE)", value=1000, step=100)

    if st.button("开始求解", type="primary", width='stretch'):
        from src.algorithm.solver import LsndSolver, SolverConfig

        config = SolverConfig(
            method=method,
            fuel_price=float(fuel_price),
            planning_horizon=int(planning_horizon),
            reject_penalty=float(reject_penalty),
            max_time=int(max_time),
            max_iterations=int(max_iterations),
            verbose=False,
        )

        progress_bar = st.progress(0, text="正在初始化求解器...")
        status_text = st.empty()

        def progress_callback(iteration, total, best_obj):
            pct = min(iteration / max(total, 1), 0.99)
            progress_bar.progress(pct, text=f"迭代 {iteration}/{total}, 当前最优: {best_obj:.0f}")

        config_with_cb = SolverConfig(
            method=method,
            fuel_price=float(fuel_price),
            planning_horizon=int(planning_horizon),
            reject_penalty=float(reject_penalty),
            max_time=int(max_time),
            max_iterations=int(max_iterations),
            verbose=False,
        )

        try:
            solver = LsndSolver(instance_name, config_with_cb, progress_callback=progress_callback)
            result = solver.solve()
            progress_bar.progress(1.0, text="求解完成!")
            st.session_state.solve_result = result
            st.success(f"求解完成! 目标值: {result.objective:,.0f}, "
                      f"航线数: {len(result.rotations)}, "
                      f"耗时: {result.solve_time:.1f}s")
        except Exception as e:
            st.error(f"求解失败: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# ========== Tab 6: 求解结果 ==========
with tab_results:
    if "solve_result" not in st.session_state or st.session_state.solve_result is None:
        st.info("尚未运行求解器。请在 '求解器' 标签页中启动求解。")
    else:
        result = st.session_state.solve_result
        st.subheader(f"求解结果 -- {result.instance_name} ({result.method})")

        # 指标概览
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("目标函数值", f"{result.objective:,.0f}")
        with col2:
            st.metric("航线数", f"{len(result.rotations)}")
        with col3:
            sr = result.flow_summary.get("service_rate", 0)
            st.metric("服务率", f"{sr:.1%}")
        with col4:
            st.metric("求解时间", f"{result.solve_time:.1f}s")
            
        # 与论文对比
        benchmarks_all = PAPER_BENCHMARKS.get(result.instance_name)
        if benchmarks_all:
            st.subheader("与论文基准结果对比 (Table 9)")
            scenario = st.selectbox("选择对比场景", list(benchmarks_all.keys()), index=list(benchmarks_all.keys()).index("Base") if "Base" in benchmarks_all else 0)
            ref = benchmarks_all[scenario]
            reject_pct = (result.flow_summary.get('rejected_ffe', 0) / result.flow_summary.get('total_demand_ffe', 1)) * 100

            comp_col1, comp_col2, comp_col3 = st.columns(3)
            with comp_col1:
                delta_z = result.objective - ref['Z_best']
                st.metric("目标函数值 (本轮 / 最优)", f"{result.objective:,.0f}", delta=f"{delta_z:,.0f} (距最优差距)", delta_color="inverse")
            with comp_col2:
                delta_r = len(result.rotations) - ref['rotations_best']
                st.metric("航线数 (本轮 / 最优)", f"{len(result.rotations)} / {ref['rotations_best']}", delta=f"{delta_r} 条", delta_color="off")
            with comp_col3:
                delta_rej = reject_pct - ref['reject_pct_best']
                st.metric("拒绝率 (本轮 / 最优)", f"{reject_pct:.1f}% / {ref['reject_pct_best']:.1f}%", delta=f"{delta_rej:.1f}%", delta_color="inverse")

            st.markdown(f"**提示**: 该实例 {scenario} 场景论文中位数为 **{ref['Z_median']:,.0f}** USD。")
            if "source" in ref:
                st.caption(f"参考来源: {ref['source']}")

        # 成本分解表 (对标论文 Table 9)
        st.subheader("成本分解 (对标论文 Table 9)")
        bd = result.cost_breakdown
        cost_data = {
            "项目": ["Z (总目标)", "Q (总收入)", "c_v (TC费用)", "c_b (燃油)",
                    "c_p (港口费)", "c_c (运河费)", "c_m (装卸费)", "c_t (转运费)",
                    "L_v (闲置船舶)", "F (拒运量 FFE)"],
            "值": [f"{bd.Z:,.0f}", f"{bd.Q:,.0f}", f"{bd.c_v:,.0f}", f"{bd.c_b:,.0f}",
                  f"{bd.c_p:,.0f}", f"{bd.c_c:,.0f}", f"{bd.c_m:,.0f}", f"{bd.c_t:,.0f}",
                  f"{bd.L_v:,.0f}", f"{bd.F:,.0f}"],
            "说明": [
                "总成本 - 总收入 - 闲置收入 + 拒运惩罚",
                "运输货物的总收入",
                "部署船舶的租金成本 (TC rate x 天数)",
                "航行燃油 + 停泊燃油",
                "固定靠港费 + 变动靠港费",
                "巴拿马 / 苏伊士运河费",
                "集装箱装卸费 (起点 + 终点)",
                "转运中转费用",
                "未使用船舶的 charter out 收入",
                "未被服务的需求总量",
            ],
        }
        st.dataframe(pd.DataFrame(cost_data), width='stretch', hide_index=True)

        # 成本结构饼图
        cost_components = {
            "TC费用": bd.c_v,
            "燃油": bd.c_b,
            "港口费": bd.c_p,
            "运河费": bd.c_c,
            "装卸费": bd.c_m,
            "转运费": bd.c_t,
        }
        positive_costs = {k: v for k, v in cost_components.items() if v > 0}
        if positive_costs:
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(positive_costs.keys()),
                values=list(positive_costs.values()),
                hole=0.4,
            )])
            fig_pie.update_layout(title="成本结构", height=400)
            st.plotly_chart(fig_pie, width='stretch')

        # 航线详情
        st.subheader("航线详情与可视化")
        if result.rotation_details:
            rot_df = pd.DataFrame(result.rotation_details)
            rot_df["port_calls"] = rot_df["port_calls"].apply(lambda x: " -> ".join(x))
            st.dataframe(rot_df, width='stretch', hide_index=True)
            
            port_coords = ports_all.set_index("UNLocode")[["Longitude", "Latitude", "name"]].to_dict("index")
            colors = px.colors.qualitative.Set2
            
            # 绘制整体网络图
            st.markdown("#### 全局规划网络地图")
            fig_all = go.Figure()
            for idx, rot_detail in enumerate(result.rotation_details):
                pc = rot_detail["port_calls"]
                color = colors[idx % len(colors)]
                for k in range(len(pc)):
                    i_p = pc[k]
                    j_p = pc[(k + 1) % len(pc)]
                    if i_p in port_coords and j_p in port_coords:
                        ic, jc = port_coords[i_p], port_coords[j_p]
                        fig_all.add_trace(go.Scattergeo(
                            lon=[ic["Longitude"], jc["Longitude"]],
                            lat=[ic["Latitude"], jc["Latitude"]],
                            mode="lines",
                            line=dict(width=2, color=color),
                            name=f"{rot_detail['id']}",
                            showlegend=(k == 0),
                        ))
                rot_lons = [port_coords[p]["Longitude"] for p in pc if p in port_coords]
                rot_lats = [port_coords[p]["Latitude"] for p in pc if p in port_coords]
                rot_names = [port_coords[p]["name"] for p in pc if p in port_coords]
                fig_all.add_trace(go.Scattergeo(
                    lon=rot_lons, lat=rot_lats, mode="markers",
                    marker=dict(size=8, color=color, line=dict(width=1, color="white")),
                    hovertext=rot_names, hoverinfo="text", showlegend=False
                ))
            fig_all.update_layout(
                geo=dict(showland=True, landcolor="rgb(243,243,243)", showcountries=True),
                margin=dict(l=0, r=0, t=0, b=0), height=500
            )
            st.plotly_chart(fig_all, width='stretch')
            
            # 单独航线可视化
            st.markdown("#### 单条航线排布可视化")
            selected_rot = st.selectbox("选择要查看的航线", [rd["id"] for rd in result.rotation_details])
            
            current_rot = next((rd for rd in result.rotation_details if rd["id"] == selected_rot), None)
            if current_rot:
                fig_single = go.Figure()
                pc = current_rot["port_calls"]
                
                # 绘制航线线段
                for k in range(len(pc)):
                    i_p = pc[k]
                    j_p = pc[(k + 1) % len(pc)]
                    if i_p in port_coords and j_p in port_coords:
                        ic, jc = port_coords[i_p], port_coords[j_p]
                        fig_single.add_trace(go.Scattergeo(
                            lon=[ic["Longitude"], jc["Longitude"]],
                            lat=[ic["Latitude"], jc["Latitude"]],
                            mode="lines",
                            line=dict(width=4, color="royalblue"),
                            showlegend=False,
                        ))
                        
                # 绘制港口点并添加序号
                rot_lons = []
                rot_lats = []
                rot_texts = []
                for idx, p in enumerate(pc):
                    if p in port_coords:
                        rot_lons.append(port_coords[p]["Longitude"])
                        rot_lats.append(port_coords[p]["Latitude"])
                        rot_texts.append(f"{idx+1}. {port_coords[p]['name']}")
                
                fig_single.add_trace(go.Scattergeo(
                    lon=rot_lons, lat=rot_lats, mode="markers+text",
                    marker=dict(size=14, color="crimson", line=dict(width=2, color="white")),
                    text=rot_texts, textposition="top center",
                    textfont=dict(size=12, color="darkred"),
                    hoverinfo="text", showlegend=False
                ))
                
                fig_single.update_layout(
                    title=f"航线: {current_rot['id']} | 船型: {current_rot['vessel_class']} ({current_rot['num_vessels']}艘) | 速度: {current_rot['speed']}kn",
                    geo=dict(showland=True, landcolor="rgb(243,243,243)", showcountries=True),
                    margin=dict(l=0, r=0, t=40, b=0), height=400
                )
                
                # 自动缩放到航线区域
                if rot_lons and rot_lats:
                    lon_center = sum(rot_lons) / len(rot_lons)
                    lat_center = sum(rot_lats) / len(rot_lats)
                    lon_range = max(rot_lons) - min(rot_lons)
                    lat_range = max(rot_lats) - min(rot_lats)
                    scale = max(lon_range, lat_range, 1)
                    if scale < 60:
                        fig_single.update_geos(center=dict(lon=lon_center, lat=lat_center), projection_scale=max(1, 150 / scale))
                        
                st.plotly_chart(fig_single, width='stretch')
        else:
            st.warning("未找到可用航线。")

        # 流量统计
        st.subheader("流量统计")
        fs = result.flow_summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总需求 (FFE/规划期)", f"{fs.get('total_demand_ffe', 0):,.0f}")
        with col2:
            st.metric("已运输 (FFE/规划期)", f"{fs.get('transported_ffe', 0):,.0f}")
        with col3:
            st.metric("被拒绝 (FFE/规划期)", f"{fs.get('rejected_ffe', 0):,.0f}")

        # 网络 KPI 指标 (对应论文 Table 11/12)
        st.subheader("网络性能指标 (对应论文 Table 11/12)")
        n_rotations = len(result.rotations)
        total_demand = fs.get('total_demand_ffe', 0)
        rejected_ffe = fs.get('rejected_ffe', 0)
        reject_pct_kpi = (rejected_ffe / total_demand * 100) if total_demand > 0 else 0

        # 计算 Dep% (船队部署比例)
        total_deployed = sum(info.get("deployed", 0) for info in result.fleet_usage.values())
        total_available = sum(info.get("available", 0) for info in result.fleet_usage.values())
        dep_pct = (total_deployed / total_available * 100) if total_available > 0 else 0

        # 计算 PCpW (每航线每周平均港口挂靠数)
        total_port_calls = sum(len(rd["port_calls"]) for rd in result.rotation_details) if result.rotation_details else 0
        pcpw = total_port_calls / max(n_rotations, 1)

        kpi_data = {
            "指标": ["Dep% (船队部署率)", "|R| (航线数)", "PCpW (平均停靠港数)", "Rej% (拒绝率)"],
            "值": [f"{dep_pct:.1f}%", f"{n_rotations}", f"{pcpw:.2f}", f"{reject_pct_kpi:.1f}%"],
            "说明": [
                "已部署船舶 / 可用船舶总数",
                "最终网络中的航线数量",
                "每条航线平均停靠港口数",
                "被拒绝运量 / 总需求量",
            ],
        }
        st.dataframe(pd.DataFrame(kpi_data), width='stretch', hide_index=True)

        # 被拒绝需求
        if result.rejected_demand:
            rejected_items = [(o, d, v) for (o, d), v in result.rejected_demand.items() if v > 0.1]
            if rejected_items:
                st.subheader(f"被拒绝需求 ({len(rejected_items)} 条)")
                rej_df = pd.DataFrame(rejected_items, columns=["起点", "终点", "拒绝量 (FFE)"])
                rej_df = rej_df.sort_values("拒绝量 (FFE)", ascending=False)
                st.dataframe(rej_df, width='stretch', height=300, hide_index=True)

        # 导出结果
        st.subheader("导出结果")
        export_data = {
            "instance": result.instance_name,
            "method": result.method,
            "status": result.status,
            "objective": result.objective,
            "solve_time": result.solve_time,
            "cost_breakdown": {
                "Z": bd.Z, "Q": bd.Q, "c_v": bd.c_v, "c_b": bd.c_b,
                "c_p": bd.c_p, "c_c": bd.c_c, "c_m": bd.c_m, "c_t": bd.c_t,
                "L_v": bd.L_v, "F": bd.F,
            },
            "flow_summary": result.flow_summary,
            "rotations": result.rotation_details,
        }
        st.download_button(
            "下载 JSON 结果",
            data=json.dumps(export_data, indent=2, ensure_ascii=False),
            file_name=f"lsnd_result_{instance_name}.json",
            mime="application/json",
        )
