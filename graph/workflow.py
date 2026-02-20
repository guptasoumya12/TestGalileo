"""LangGraph workflow definition with Galileo Observe integration."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from langgraph.graph import END, StateGraph

from agents.quality_check import quality_check_agent
from agents.enhancement import enhancement_agent
from agents.video_generation import video_generation_agent
from graph.state import ImagePipelineState

# ── Optional Galileo imports ────────────────────────────────────────
# If galileo-observe is not installed or credentials are missing we
# still let the pipeline run without observability.

try:
    from galileo_observe import ObserveWorkflows
    _GALILEO_AVAILABLE = True
except ImportError:
    _GALILEO_AVAILABLE = False

# Module-level holder shared across all nodes during a single run.
_observe_workflows: Optional[Any] = None
_galileo_init_done = False


def _init_galileo() -> Optional[Any]:
    """Initialise the ObserveWorkflows tracker (singleton per run).

    Returns the ObserveWorkflows instance or None if unavailable.
    """
    global _observe_workflows, _galileo_init_done  # noqa: PLW0603
    if _galileo_init_done:
        return _observe_workflows

    _galileo_init_done = True

    if not _GALILEO_AVAILABLE:
        print("⚠️  galileo-observe not installed – running without observability.")
        return None

    api_key = os.getenv("GALILEO_API_KEY")
    project = os.getenv("GALILEO_PROJECT", "TestGalileo")

    if not api_key:
        print("⚠️  GALILEO_API_KEY not set – running without observability.")
        return None

    try:
        _observe_workflows = ObserveWorkflows(project_name=project)
        print(f"✅ Galileo Observe initialised (project={project!r})")
    except Exception as exc:
        print(f"⚠️  Failed to initialise Galileo Observe: {exc}")
        print("   Continuing without observability.")
        return None

    return _observe_workflows


def _start_galileo_workflow(input_summary: str) -> None:
    """Begin a new top-level agent workflow in Galileo."""
    ow = _init_galileo()
    if ow is None:
        return
    try:
        ow.add_agent_workflow(
            input=input_summary,
            name="image-enhancement-pipeline",
            metadata={"framework": "langgraph"},
        )
        print("\n📤 [Galileo] Started workflow: image-enhancement-pipeline")
        print(f"   Input: {input_summary[:200]}{'...' if len(input_summary) > 200 else ''}")
    except Exception as exc:
        print(f"⚠️  Galileo: failed to start workflow – {exc}")


def _log_galileo_step(node_name: str, input_text: str, output_text: str, duration_ns: int) -> None:
    """Log a single tool step inside the current Galileo workflow."""
    ow = _init_galileo()
    if ow is None:
        return
    try:
        ow.add_tool_step(
            input=input_text,
            output=output_text,
            name=node_name,
            duration_ns=duration_ns,
            status_code=200,
            metadata={"node": node_name},
        )
        duration_ms = duration_ns / 1_000_000
        print(f"\n📤 [Galileo] Logged step: {node_name} ({duration_ms:.0f}ms)")
        print(f"   Input:  {input_text[:150]}{'...' if len(input_text) > 150 else ''}")
        print(f"   Output: {output_text[:150]}{'...' if len(output_text) > 150 else ''}")
    except Exception as exc:
        print(f"⚠️  Galileo: failed to log step '{node_name}' – {exc}")


def _conclude_and_upload(output_summary: str, duration_ns: int) -> None:
    """Conclude the Galileo workflow and upload all recorded data."""
    ow = _init_galileo()
    if ow is None:
        return
    try:
        duration_ms = duration_ns / 1_000_000
        ow.conclude_workflow(
            output=output_summary,
            duration_ns=duration_ns,
            status_code=200,
        )
        print(f"\n📤 [Galileo] Concluded workflow (total {duration_ms:.0f}ms)")
        print(f"   Output: {output_summary[:200]}{'...' if len(output_summary) > 200 else ''}")
        print(f"📤 [Galileo] Uploading workflows to Galileo...")
        results = ow.upload_workflows()
        print(f"✅ Galileo: uploaded {len(results)} workflow(s) successfully.")
    except Exception as exc:
        print(f"⚠️  Galileo: failed to upload workflows – {exc}")


# ── Galileo-instrumented wrapper helpers ────────────────────────────

def _observed(node_name: str, fn):
    """Wrap a LangGraph node function with Galileo step logging."""

    def wrapper(state: ImagePipelineState) -> dict:
        input_summary = json.dumps(
            {k: v for k, v in state.items() if k != "enhancement_log"},
            default=str,
        )

        start_ns = time.time_ns()
        try:
            result = fn(state)
        except Exception as exc:
            # Log error step to Galileo if possible
            duration_ns = time.time_ns() - start_ns
            _log_galileo_step(node_name, input_summary, f"ERROR: {exc}", duration_ns)
            raise

        duration_ns = time.time_ns() - start_ns
        output_summary = json.dumps(result, default=str)
        _log_galileo_step(node_name, input_summary, output_summary, duration_ns)
        return result

    wrapper.__name__ = node_name
    return wrapper


# ── Routing logic ───────────────────────────────────────────────────

def _should_continue(state: ImagePipelineState) -> str:
    """Conditional edge: decide whether to enhance again or generate video."""
    if state.get("quality_passed", False):
        return "generate_video"
    if state.get("enhancement_iteration", 0) >= state.get("max_iterations", 5):
        print("\n⚠️  Max enhancement iterations reached – proceeding to video generation.")
        return "generate_video"
    return "enhance"


# ── Graph builder ───────────────────────────────────────────────────

def build_workflow() -> StateGraph:
    """Construct and return the compiled LangGraph workflow.

    Graph topology
    ==============
    START ──► quality_check ──► should_enhance? ──┬─► enhance ──► quality_check (loop)
                                                   └─► generate_video ──► END
    """
    graph = StateGraph(ImagePipelineState)

    # Register nodes (wrapped with Galileo instrumentation)
    graph.add_node("quality_check", _observed("quality_check", quality_check_agent))
    graph.add_node("enhance", _observed("enhance", enhancement_agent))
    graph.add_node("generate_video", _observed("generate_video", video_generation_agent))

    # Edges
    graph.set_entry_point("quality_check")

    graph.add_conditional_edges(
        "quality_check",
        _should_continue,
        {
            "enhance": "enhance",
            "generate_video": "generate_video",
        },
    )

    graph.add_edge("enhance", "quality_check")
    graph.add_edge("generate_video", END)

    return graph.compile()


def run_pipeline(initial_state: ImagePipelineState) -> ImagePipelineState:
    """Build the graph, run it, and handle Galileo workflow lifecycle.

    This is the recommended entry-point – it ensures the Galileo workflow
    is properly started, concluded, and uploaded.
    """
    app = build_workflow()

    # Open the Galileo workflow
    input_summary = json.dumps(
        {k: v for k, v in initial_state.items() if k != "enhancement_log"},
        default=str,
    )
    _start_galileo_workflow(input_summary)

    pipeline_start = time.time_ns()
    final_state = app.invoke(initial_state)
    pipeline_duration = time.time_ns() - pipeline_start

    # Close & upload
    output_summary = json.dumps(
        {k: v for k, v in final_state.items() if k != "enhancement_log"},
        default=str,
    )
    _conclude_and_upload(output_summary, pipeline_duration)

    return final_state
