import os
from pathlib import Path
import pandas as pd
import streamlit as st
from agents import NormalizedData, DataAgent, SummaryAgent, RouterAgent

st.set_page_config(page_title="Mini Multi-Agent ChatBot", layout="wide")
st.title("Multi-Agent ChatBot")
st.caption("Small multi-agent assignment for all the CSVs. One agent does numbers, another does text. Making space for the Magician!!!")

if "chat" not in st.session_state:
    st.session_state["chat"] = []

def clear_chat():
    st.session_state["chat"] = []

DEFAULT_FILES = [
    "/Users/pawanparankusam/Downloads/multi_agent_sales_demo/data/Amazon-Sale-Report.csv",
    "/Users/pawanparankusam/Downloads/multi_agent_sales_demo/data/Cloud-Warehouse-Compersion-Chart.csv",
    "/Users/pawanparankusam/Downloads/multi_agent_sales_demo/data/Expense-IIGF.csv",
    "/Users/pawanparankusam/Downloads/multi_agent_sales_demo/data/International-sale-Report.csv",
    "/Users/pawanparankusam/Downloads/multi_agent_sales_demo/data/May-2022.csv",
    "/Users/pawanparankusam/Downloads/multi_agent_sales_demo/data/P-L-March-2021.csv",
    "/Users/pawanparankusam/Downloads/multi_agent_sales_demo/data/Sale-Report.csv",
]

available = [p for p in DEFAULT_FILES if os.path.exists(p)]
left, right = st.columns([1, 2], vertical_alignment="top")

with left:
    st.subheader("Data")
    mode = st.radio("Choose input", ["Use preset CSV", "Upload CSV"], index=0)

    if mode == "Use preset CSV":
        if not available:
            st.warning("No preset CSVs found.")
            selected_path = None
        else:
            file_name_to_path = {Path(p).name: p for p in available}
            selected_name = st.selectbox("Select one file", list(file_name_to_path.keys()), index=0)
            selected_path = file_name_to_path[selected_name]
        uploaded = None
    else:
        uploaded = st.file_uploader("Upload a CSV", type=["csv"])
        selected_path = None

    current_file_key = None
    if mode == "Use preset CSV" and selected_path:
        current_file_key = str(selected_path)
    elif mode == "Upload CSV" and uploaded is not None:
        current_file_key = uploaded.name

    if "active_file_key" not in st.session_state:
        st.session_state["active_file_key"] = current_file_key

    if current_file_key and st.session_state["active_file_key"] != current_file_key:
        st.session_state["active_file_key"] = current_file_key
        clear_chat()

    st.subheader("Quick actions")
    run_summary = st.button("Generate quick summary")
    show_sample = st.checkbox("Preview data", value=True)

    if st.button("Clear Chat"):
        clear_chat()
        st.rerun()

#Loading data
sales = None
try:
    if mode == "Use preset CSV" and selected_path:
        sales = NormalizedData.from_csv(selected_path)
        st.caption(f"Using file: {Path(selected_path).name}")
    elif mode == "Upload CSV" and uploaded is not None:
        tmp_path = Path(uploaded.name)
        tmp_path.write_bytes(uploaded.getbuffer())
        st.session_state["tmp_path"] = str(tmp_path)
        sales = NormalizedData.from_csv(str(tmp_path))
except Exception as e:
    st.error(f"Could not load file: {e}")

if sales is None:
    st.stop()

data_agent = DataAgent(sales)
summary_agent = SummaryAgent(sales)
router = RouterAgent(data_agent, summary_agent)

with right:
    st.subheader(f"Loaded: {sales.source_name}")

    if show_sample:
        st.dataframe(sales.df.head(25), use_container_width=True)

    if run_summary:
        agent_name, resp = ("SummaryAgent", summary_agent.run(style="short"))
        st.success(f"{agent_name}: {resp}")

    st.divider()
    st.subheader("Chat")

    if "chat" not in st.session_state:
        st.session_state["chat"] = []

    for role, msg in st.session_state["chat"]:
        with st.chat_message(role):
            st.write(msg)

    user_msg = st.chat_input("Ask something like: 'Total revenue', 'Top product', 'Give me a summary'")
    if user_msg:
        st.session_state["chat"].append(("user", user_msg))
        agent_name, resp = router.route(user_msg)
        st.session_state["chat"].append(("assistant", f"[{agent_name}] {resp}"))

        with st.chat_message("assistant"):
            st.write(f"[{agent_name}] {resp}")


