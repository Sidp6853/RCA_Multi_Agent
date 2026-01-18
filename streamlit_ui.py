import streamlit as st
import requests
import json
from pathlib import Path

# -----------------------
# CONFIG
# -----------------------

API_BASE_URL = "http://127.0.0.1:8000"
ANALYZE_ENDPOINT = f"{API_BASE_URL}/analyze"
RESULTS_ENDPOINT = f"{API_BASE_URL}/results"

# -----------------------
# PAGE CONFIG
# -----------------------

st.set_page_config(
    page_title="Multi-Agent RCA System",
    layout="wide"
)

st.title("🛠 Multi-Agent RCA + Fix + Patch System")

st.caption("Root Cause Analysis → Fix Suggestion → Patch Generation")

# -----------------------
# INPUT SECTION
# -----------------------

st.subheader("📥 Upload Error Trace")

uploaded_file = st.file_uploader(
    "Upload trace file (.json / .txt)",
    type=["json", "txt"]
)

trace_payload = {}

if uploaded_file:
    trace_text = uploaded_file.read().decode("utf-8")
    trace_payload["trace_text"] = trace_text

# -----------------------
# EXECUTE BUTTON
# -----------------------

run_clicked = st.button("🚀 Run RCA Workflow")

# -----------------------
# EXECUTION
# -----------------------

if run_clicked:

    if not trace_payload:
        st.warning("Please upload a trace file first")
    else:

        with st.spinner("⏳ Agents are working..."):

            try:
                response = requests.post(
                    ANALYZE_ENDPOINT,
                    json=trace_payload,
                    timeout=300
                )

                if response.status_code != 200:
                    st.error(response.text)
                else:

                    result = response.json()

                    if not result["success"]:
                        st.error(result["error"])
                    else:

                        st.success("✅ Workflow Completed")

                        data = result["data"]

                        # -----------------------
                        # SUMMARY PANEL
                        # -----------------------

                        st.subheader("📊 Execution Summary")

                        col1, col2, col3 = st.columns(3)

                        col1.metric("Status", data["workflow_status"])
                        col2.metric("Messages Logged", data["message_count"])
                        col3.metric("Patch Generated", "Yes" if data["patch"] else "No")

                        # -----------------------
                        # CORE OUTPUT VIEW
                        # -----------------------

                        st.divider()
                        st.subheader("📂 Agent Outputs")

                        tab1, tab2, tab3 = st.tabs(
                            ["🔍 RCA Output", "🧩 Fix Plan", "🩹 Patch Result"]
                        )

                        with tab1:
                            if data["rca"]:
                                st.json(data["rca"])
                            else:
                                st.info("No RCA output")

                        with tab2:
                            if data["fix"]:
                                st.json(data["fix"])
                            else:
                                st.info("No Fix output")

                        with tab3:
                            patch_data = data.get("patch")

                            if patch_data and patch_data.get("success"):

                                st.json(patch_data)

                                patch_path = patch_data.get("absolute_path")

                                if patch_path and Path(patch_path).exists():
                                    with open(patch_path, "rb") as f:
                                        st.download_button(
                                            "⬇ Download Patch File",
                                            data=f,
                                            file_name=Path(patch_path).name,
                                            mime="text/plain"
                                        )
                            else:
                                st.info("No patch generated")

            except Exception as e:
                st.error(str(e))

# -----------------------
# MESSAGE HISTORY + SHARED MEMORY VIEW
# -----------------------

st.divider()
st.subheader("📜 System Logs & Shared Memory")

view_choice = st.radio(
    "Select View",
    ["Shared Memory", "Message History"],
    horizontal=True
)

if st.button("🔄 Load Latest From Server"):

    try:
        res = requests.get(RESULTS_ENDPOINT)

        if res.status_code == 200:

            payload = res.json()

            if payload["success"]:

                results = payload["results"]

                if view_choice == "Shared Memory":
                    memory = results.get("shared_memory")
                    if memory:
                        st.json(memory)
                    else:
                        st.warning("Shared memory not found")

                else:
                    history = results.get("message_history")
                    if history:
                        st.json(history)
                    else:
                        st.warning("Message history not found")

            else:
                st.warning("No saved results found")

        else:
            st.error("Server error")

    except Exception as e:
        st.error(str(e))
