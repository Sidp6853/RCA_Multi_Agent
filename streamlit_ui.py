import streamlit as st
import requests
import json
import os
from pathlib import Path

st.set_page_config(
    page_title="Multi-Agent RCA System",
    page_icon="🔍",
    layout="wide"
)

# API Configuration
API_URL = "http://localhost:8000/analyze"
HEALTH_URL = "http://localhost:8000/health"



st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🔍 Multi-Agent RCA System</h1>', unsafe_allow_html=True)
st.markdown("### Automated Root Cause Analysis, Fix Suggestion & Patch Generation")

# Check API health
with st.sidebar:
    st.header("⚙️ System Status")
    
    try:
        health_response = requests.get(HEALTH_URL, timeout=2)
        if health_response.status_code == 200:
            st.success("✅ API Server: Online")
        else:
            st.error("❌ API Server: Error")
    except:
        st.error("❌ API Server: Offline")
        st.warning("Please start the API server:\n\n`python api.py`")
    
    st.markdown("---")
    st.info("**How it works:**\n\n1. Upload trace file path\n2. Specify codebase path\n3. Click Analyze\n4. Get results!")

# Main content
st.markdown("---")

# Input form
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Trace File")
    trace_file_path = st.text_input(
        "Path to error trace JSON file",
        help="Full or relative path to your error trace file"
    )
    
    # Show file preview if exists
    if trace_file_path and os.path.exists(trace_file_path):
        st.success(f"✅ File found: {trace_file_path}")
        file_size = os.path.getsize(trace_file_path)
        st.caption(f"Size: {file_size:,} bytes")
    elif trace_file_path:
        st.warning(f"⚠️ File not found: {trace_file_path}")

with col2:
    st.subheader("📁 Codebase")
    codebase_root = st.text_input(
        "Path to codebase directory",
         help="Root directory of your codebase to analyze"
    )
    
    # Show directory preview if exists
    if codebase_root and os.path.exists(codebase_root):
        st.success(f"✅ Directory found: {codebase_root}")
        try:
            file_count = len(list(Path(codebase_root).rglob("*.py")))
            st.caption(f"Python files: {file_count}")
        except:
            pass
    elif codebase_root:
        st.warning(f"⚠️ Directory not found: {codebase_root}")

st.markdown("---")

# Analyze button
analyze_button = st.button("🚀 Start Analysis", type="primary", use_container_width=True)

# Results section
if analyze_button:
    # Validate inputs
    if not trace_file_path or not codebase_root:
        st.error("❌ Please provide both trace file and codebase paths")
    elif not os.path.exists(trace_file_path):
        st.error(f"❌ Trace file not found: {trace_file_path}")
    elif not os.path.exists(codebase_root):
        st.error(f"❌ Codebase directory not found: {codebase_root}")
    else:
        # Show loading spinner
        with st.spinner("🔄 Running multi-agent workflow... This may take a few minutes."):
            try:
                # Prepare request
                payload = {
                    "trace_file_path": trace_file_path,
                    "codebase_root": codebase_root
                }
                
                # Call API
                response = requests.post(
                    API_URL,
                    json=payload,
                    timeout=600  # 10 minute timeout
                )
                
                # Parse response
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get("success"):
                        st.balloons()
                        st.success("✅ **Analysis completed successfully!**")
                        
                        # Display results in tabs
                        tab1, tab2, tab3 = st.tabs(["🔍 RCA Results", "🔧 Fix Plan", "📝 Patch"])
                        
                        with tab1:
                            st.subheader("Root Cause Analysis")
                            rca = result["results"]["rca"]
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Error Type", rca.get("error_type", "N/A"))
                                st.metric("Affected Line", rca.get("affected_line", "N/A"))
                            with col2:
                                st.metric("Affected File", rca.get("affected_file", "N/A"))
                                st.metric("Iterations", rca.get("iterations", "N/A"))
                            
                            st.markdown("**Error Message:**")
                            st.code(rca.get("error_message", "N/A"), language="text")
                            
                            st.markdown("**Root Cause:**")
                            st.info(rca.get("root_cause", "N/A"))
                        
                        with tab2:
                            st.subheader("Fix Suggestion")
                            fix = result["results"]["fix"]
                            
                            st.markdown("**Fix Summary:**")
                            st.success(fix.get("fix_summary", "N/A"))
                            
                            st.markdown("**Files to Modify:**")
                            files = fix.get("files_to_modify", [])
                            if files:
                                for file in files:
                                    st.code(file)
                            else:
                                st.write("No files to modify")
                            
                            st.markdown("**Patch Plan:**")
                            patch_plan = fix.get("patch_plan", [])
                            if patch_plan:
                                for i, step in enumerate(patch_plan, 1):
                                    st.write(f"{i}. {step}")
                            else:
                                st.write("No patch plan available")
                            
                            st.markdown("**Safety Considerations:**")
                            st.info(fix.get("safety_considerations", "N/A"))
                            
                            st.caption(f"Iterations: {fix.get('iterations', 'N/A')}")
                        
                        with tab3:
                            st.subheader("Patch Generation")
                            patch = result["results"]["patch"]
                            
                            if patch.get("success"):
                                st.success("✅ **Patch created successfully!**")
                                
                                st.markdown("**Patch File Location:**")
                                patch_file = patch.get("patch_file", "N/A")
                                st.code(patch_file, language="text")
                                
                                st.markdown("**Original File:**")
                                st.code(patch.get("original_file", "N/A"), language="text")
                                
                                # Big success message
                                st.markdown("---")
                                st.markdown(f"""
                                <div class="success-box">
                                    <h3>🎉 Patch Generated!</h3>
                                    <p>Find your patch file at:</p>
                                    <code>{patch_file}</code>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.error("❌ Patch generation failed")
                            
                            st.caption(f"Iterations: {patch.get('iterations', 'N/A')}")
                        
                        # Output files section
                        st.markdown("---")
                        st.subheader("📁 Output Files")
                        output_files = result.get("output_files", {})
                        if output_files:
                            cols = st.columns(len(output_files))
                            for i, (name, path) in enumerate(output_files.items()):
                                with cols[i]:
                                    st.markdown(f"**{name}**")
                                    st.code(path, language="text")
                        else:
                            st.write("No output files generated")
                    
                    else:
                        st.error(f"❌ **Analysis failed:** {result.get('message', 'Unknown error')}")
                        if result.get("error"):
                            st.code(result["error"], language="text")
                
                else:
                    st.error(f"❌ **API Error (Status {response.status_code})**")
                    st.code(response.text, language="json")
            
            except requests.exceptions.Timeout:
                st.error("⏱️ **Request timed out**")
                st.info("The workflow is taking longer than expected. Check the API server logs for progress.")
            
            except requests.exceptions.ConnectionError:
                st.error("🔌 **Connection Error**")
                st.warning("Cannot connect to API server. Make sure it's running:\n\n`python api.py`")
            
            except Exception as e:
                st.error(f"❌ **Unexpected Error:** {str(e)}")
                st.code(str(e), language="text")

