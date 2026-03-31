"""
LINERLIB 班轮航运网络可视化应用
使用 Streamlit + Plotly 实现交互式可视化。
启动: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from data_reader import load_instance, load_distances, INSTANCES

st.set_page_config(page_title="LINERLIB 航运网络可视化", layout="wide")
st.title("LINERLIB 班轮航运网络可视化")

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
tab_map, tab_demand, tab_fleet, tab_ports = st.tabs(["网络地图", "需求分析", "船队信息", "港口详情"])

# ========== Tab 1: 网络地图 ==========
with tab_map:
    # 构建港口坐标映射
    port_coords = ports_all.set_index("UNLocode")[["Longitude", "Latitude", "name"]].to_dict("index")

    fig = go.Figure()

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
                    text=f"{oc['name']} → {dc['name']}<br>FFE/周: {row['FFEPerWeek']}<br>收入: {row['Revenue_1']}",
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

    # 根据实例自动调整视图范围
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

    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"当前显示 {len(demand_filtered)} 条航线（按 FFE/周 排序 Top {top_n_routes}）")

# ========== Tab 2: 需求分析 ==========
with tab_demand:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("需求分布 (FFE/周)")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=demand["FFEPerWeek"], nbinsx=40, marker_color="steelblue"))
        fig_hist.update_layout(xaxis_title="FFE/周", yaxis_title="航线数", height=350)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.subheader("收入 vs 运量")
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=demand["FFEPerWeek"], y=demand["Revenue_1"],
            mode="markers", marker=dict(size=5, opacity=0.6, color="teal"),
            hovertext=demand.apply(lambda r: f"{r['Origin']}→{r['Destination']}", axis=1),
            hoverinfo="text+x+y",
        ))
        fig_scatter.update_layout(xaxis_title="FFE/周", yaxis_title="单箱收入", height=350)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # 港口吞吐量排名
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
    st.plotly_chart(fig_bar, use_container_width=True)

    # 需求数据表
    st.subheader("需求数据明细")
    st.dataframe(demand.sort_values("FFEPerWeek", ascending=False), use_container_width=True, height=300)

# ========== Tab 3: 船队信息 ==========
with tab_fleet:
    st.subheader(f"{instance_name} 实例 — 可用船队")
    st.dataframe(fleet, use_container_width=True)

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
        st.plotly_chart(fig_fleet, use_container_width=True)

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
        st.plotly_chart(fig_tc, use_container_width=True)

    # 总运力统计
    total_capacity = (fleet["Capacity FFE"] * fleet["Quantity"]).sum()
    total_vessels = fleet["Quantity"].sum()
    total_demand = demand["FFEPerWeek"].sum()
    st.metric("总运力 (FFE)", f"{total_capacity:,.0f}", delta=f"供需比: {total_capacity/total_demand:.2f}" if total_demand > 0 else None)
    st.metric("船舶总数", f"{total_vessels}")

# ========== Tab 4: 港口详情 ==========
with tab_ports:
    st.subheader("实例涉及港口")
    st.dataframe(ports.sort_values("name"), use_container_width=True, height=400)

    st.subheader("全部港口数据库")
    search = st.text_input("搜索港口 (名称或代码)")
    display_ports = ports_all
    if search:
        mask = (
            ports_all["name"].str.contains(search, case=False, na=False) |
            ports_all["UNLocode"].str.contains(search, case=False, na=False)
        )
        display_ports = ports_all[mask]
    st.dataframe(display_ports.sort_values("name"), use_container_width=True, height=400)
    st.caption(f"共 {len(display_ports)} 个港口")
