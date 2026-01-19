import os
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from app.workflow import pipeline, PipelineState

# Load environment variables
load_dotenv()


def run_rca_workflow(trace_file_path: str, codebase_root: str):
    """
    Run the complete multi-agent RCA workflow
    
    Args:
        trace_file_path: Path to error trace JSON file
        codebase_root: Root directory of buggy codebase
    """
    
    # Set environment variable for tools
    os.environ["CODEBASE_ROOT"] = codebase_root
    
    print("=" * 80)
    print("🔍 Multi-Agent RCA System")
    print("=" * 80)
    print(f"Trace File: {trace_file_path}")
    print(f"Codebase Root: {codebase_root}")
    print("=" * 80)
    
    # Validate inputs
    if not os.path.exists(trace_file_path):
        print(f" Error: Trace file not found: {trace_file_path}")
        return
    
    if not os.path.exists(codebase_root):
        print(f" Error: Codebase directory not found: {codebase_root}")
        return
    
    # Load error trace
    print("\n Loading error trace...")
    with open(trace_file_path, "r", encoding="utf-8") as f:
        trace_data = f.read()
    print("✅ Trace loaded successfully")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f" Output directory: {output_dir}")
    
    # Prepare initial state
    print("\n Starting workflow pipeline...")
    initial_state = PipelineState()
    initial_state.messages = [HumanMessage(content=trace_data)]
    
    # Configure LangGraph with thread ID for checkpointing
    thread_id = f"{timestamp}_{hash(trace_file_path) % 10000}"
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Run the workflow
        print("\n" + "=" * 80)
        final_state = pipeline.invoke(initial_state, config=config)
        final_state = dict(final_state)
        print("=" * 80)
        
        # Get complete message history from checkpoint
        history_state = pipeline.get_state(config)
        full_message_history = history_state.values["messages"]
        
        # Check for errors
        if final_state.get("error"):
            print(f"\n Workflow failed: {final_state['error']}")
            return
        
        # Print results
        print("\n" + "=" * 80)
        print(" WORKFLOW COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        # 1. RCA Results
        print("\nRCA RESULTS:")
        print("-" * 80)
        rca_result = final_state.get("rca_result")
        if rca_result:
            print(f"Error Type: {rca_result.get('error_type')}")
            print(f"Error Message: {rca_result.get('error_message')}")
            print(f"Affected File: {rca_result.get('affected_file')}")
            print(f"Affected Line: {rca_result.get('affected_line')}")
            print(f"\nRoot Cause:\n{rca_result.get('root_cause')}")
        else:
            print(" No RCA result available")
        
        # 2. Fix Results
        print("\n FIX PLAN:")
        print("-" * 80)
        fix_result = final_state.get("fix_result")
        if fix_result:
            print(f"Summary: {fix_result.get('fix_summary')}")
            print(f"Files to Modify: {', '.join(fix_result.get('files_to_modify', []))}")
            print(f"\nStep-by-Step Plan:")
            for i, step in enumerate(fix_result.get('patch_plan', []), 1):
                print(f"  {i}. {step}")
            print(f"\nSafety Considerations: {fix_result.get('safety_considerations')}")
        else:
            print(" No fix plan available")
        
        # 3. Patch Results
        print("\n PATCH GENERATION:")
        print("-" * 80)
        patch_result = final_state.get("patch_result")
        if patch_result and patch_result.get("success"):
            print(f"Patch created successfully!")
            print(f"Patch File: {patch_result.get('patch_file')}")
            print(f"Original File: {patch_result.get('original_file')}")
            print(f"Size: {patch_result.get('size_bytes', 0)} bytes")
            print(f"Lines: {patch_result.get('lines', 0)}")
        else:
            print(" Patch generation failed")
        
       
        print("\n" + "=" * 80)
        print(" SAVING OUTPUTS")
        print("=" * 80)
        
        
        message_history = {
            "thread_id": thread_id,
            "timestamp": datetime.now().isoformat(),
            "input": {
                "trace_file": trace_file_path,
                "codebase_root": codebase_root
            },
            "complete_message_history": [
                {
                    "role": msg.type if hasattr(msg, 'type') else getattr(msg, 'role', 'unknown'),
                    "content": msg.content[:500] + "..." if len(str(msg.content)) > 500 else msg.content,
                    "tool_calls": getattr(msg, "tool_calls", []),
                    "name": getattr(msg, "name", None)
                } for msg in full_message_history
            ],
            "rca_result": final_state.get("rca_result"),
            "fix_result": final_state.get("fix_result"),
            "patch_result": final_state.get("patch_result"),
            "stats": {
                "total_messages": len(full_message_history),
                "tool_calls": len([m for m in full_message_history if getattr(m, "tool_calls", [])]),
                "success": bool(patch_result and patch_result.get("success"))
            }
        }
        
        message_history_path = output_dir / "message_history.json"
        with open(message_history_path, "w", encoding="utf-8") as f:
            json.dump(message_history, f, indent=2, default=str)
        print(f" Message history saved: {message_history_path}")
        
        # 2. Save shared memory state
        shared_memory = {
            "rca_result": final_state.get("rca_result"),
            "fix_result": final_state.get("fix_result"),
            "patch_result": final_state.get("patch_result")
        }
        
        shared_memory_path = output_dir / "shared_memory.json"
        with open(shared_memory_path, "w", encoding="utf-8") as f:
            json.dump(shared_memory, f, indent=2, default=str)
        print(f" Shared memory saved: {shared_memory_path}")
        
        # Summary
        print("\n" + "=" * 80)
        print(" SUMMARY")
        print("=" * 80)
        print(f"Total Messages: {len(full_message_history)}")
        print(f"Tool Calls: {len([m for m in full_message_history if getattr(m, 'tool_calls', [])])}")
        print(f"Patch File: {patch_result.get('patch_file') if patch_result else 'N/A'}")
        print(f"Output Directory: {output_dir}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n Error during workflow execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    
    TRACE_FILE = "trace_1.json"  
    CODEBASE_ROOT = "codebase"   
    
    
    if not os.getenv("GOOGLE_API_KEY"):
        print(" Error: GOOGLE_API_KEY not set in environment")
        print("Please set it in .env file or export it:")
        print("  export GOOGLE_API_KEY='your-api-key'")
        exit(1)
    
    
    run_rca_workflow(TRACE_FILE, CODEBASE_ROOT)