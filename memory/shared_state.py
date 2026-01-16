from typing_extensions import TypedDict
from typing import Optional, Dict, Any
from langgraph_swarm import SwarmState


class RCAState(SwarmState):
    rca: Optional[Dict[str, Any]]
    fix_plan: Optional[Dict[str, Any]]
    patch_metadata: Optional[Dict[str, Any]]
