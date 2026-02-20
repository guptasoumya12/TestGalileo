"""LangGraph workflow definition with Galileo Observe integration."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
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
_galileo_lock = threading.Lock()  # serialise Galileo API access


def _init_galileo() -> Optional[Any]:
    """Initialise the ObserveWorkflows tracker (singleton per process).

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


# ── Thread-local step collector ─────────────────────────────────────
# Each thread accumulates steps in its own buffer during pipeline
# execution.  Once the pipeline finishes, the entire workflow
# (start → steps → conclude) is replayed into the Galileo API under
# a single lock – guaranteeing no interleaving between threads.


@dataclass
class _StepRecord:
    node_name: str
    input_text: str
    output_text: str
    duration_ns: int
    status_code: int = 200  # 200 = ok, 500 = error


_thread_local = threading.local()


def _get_step_buffer() -> list[_StepRecord]:
    """Return the step buffer for the current thread (create if needed)."""
    if not hasattr(_thread_local, "steps"):
        _thread_local.steps = []
    return _thread_local.steps


def _collect_step(
    node_name: str,
    input_text: str,
    output_text: str,
    duration_ns: int,
    *,
    status_code: int = 200,
) -> None:
    """Buffer a step record (thread-local, no lock needed)."""
    buf = _get_step_buffer()
    buf.append(_StepRecord(node_name, input_text, output_text, duration_ns, status_code))
    duration_ms = duration_ns / 1_000_000
    tag = "ERROR" if status_code >= 400 else "OK"
    print(f"  📝 [Collected] step: {node_name} ({duration_ms:.0f}ms) [{tag}]")


def _replay_workflow_to_galileo(
    input_summary: str,
    output_summary: str,
    total_duration_ns: int,
    *,
    workflow_status_code: int = 200,
) -> None:
    """Replay all collected steps as a single Galileo workflow (atomic).

    Args:
        workflow_status_code: HTTP-style status for the overall workflow.
            200 = success, 500 = pipeline crashed.
    """
    ow = _init_galileo()
    if ow is None:
        return

    steps = _get_step_buffer()
    has_errors = any(s.status_code >= 400 for s in steps)
    wf_code = workflow_status_code if workflow_status_code >= 400 else (500 if has_errors else 200)

    with _galileo_lock:
        try:
            # 1. Start workflow
            ow.add_agent_workflow(
                input=input_summary,
                name="image-enhancement-pipeline",
                metadata={
                    "framework": "langgraph",
                    "has_errors": str(has_errors),
                    "workflow_status": str(wf_code),
                },
            )
            # 2. Replay every step (with per-step status codes)
            for step in steps:
                ow.add_tool_step(
                    input=step.input_text,
                    output=step.output_text,
                    name=step.node_name,
                    duration_ns=step.duration_ns,
                    status_code=step.status_code,
                    metadata={
                        "node": step.node_name,
                        "status": "error" if step.status_code >= 400 else "ok",
                    },
                )
            # 3. Conclude
            ow.conclude_workflow(
                output=output_summary,
                duration_ns=total_duration_ns,
                status_code=wf_code,
            )
            total_ms = total_duration_ns / 1_000_000
            status_tag = "❌ ERROR" if wf_code >= 400 else "✅ OK"
            print(f"  📤 [Galileo] Replayed workflow ({len(steps)} steps, {total_ms:.0f}ms) [{status_tag}]")
        except Exception as exc:
            print(f"  ⚠️  Galileo: failed to replay workflow – {exc}")

    # Clear the buffer for this thread
    steps.clear()


def flush_galileo() -> int:
    """Upload all accumulated workflows to Galileo. Returns count uploaded."""
    ow = _init_galileo()
    if ow is None:
        return 0
    with _galileo_lock:
        try:
            print(f"📤 [Galileo] Uploading workflows to Galileo...")
            results = ow.upload_workflows()
            print(f"✅ Galileo: uploaded {len(results)} workflow(s) successfully.")
            return len(results)
        except Exception as exc:
            print(f"⚠️  Galileo: failed to upload workflows – {exc}")
            return 0


def reset_galileo() -> None:
    """Reset the Galileo singleton so a fresh ObserveWorkflows is created."""
    global _observe_workflows, _galileo_init_done  # noqa: PLW0603
    _observe_workflows = None
    _galileo_init_done = False


# ── Galileo-instrumented wrapper helpers ────────────────────────────

def _observed(node_name: str, fn):
    """Wrap a LangGraph node function with step collection."""

    def wrapper(state: ImagePipelineState) -> dict:
        input_summary = json.dumps(
            {k: v for k, v in state.items() if k != "enhancement_log"},
            default=str,
        )

        start_ns = time.time_ns()
        try:
            result = fn(state)
        except Exception as exc:
            duration_ns = time.time_ns() - start_ns
            _collect_step(
                node_name,
                input_summary,
                f"ERROR: {type(exc).__name__}: {exc}",
                duration_ns,
                status_code=500,
            )
            raise

        duration_ns = time.time_ns() - start_ns
        output_summary = json.dumps(result, default=str)
        _collect_step(node_name, input_summary, output_summary, duration_ns)
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


def run_pipeline(
    initial_state: ImagePipelineState,
    *,
    defer_upload: bool = False,
    raise_on_error: bool = True,
) -> ImagePipelineState:
    """Build the graph, run it, and handle Galileo workflow lifecycle.

    Args:
        initial_state: The initial pipeline state dict.
        defer_upload: If True, replay steps into Galileo but do NOT upload
                      yet.  Call ``flush_galileo()`` later to push all
                      accumulated workflows in one batch.
        raise_on_error: If False, catch pipeline exceptions and return the
                        last known state (useful for anomaly testing).

    This is the recommended entry-point – it ensures the Galileo workflow
    is properly started, concluded, and (optionally) uploaded even on errors.
    """
    # Clear any leftover step buffer for this thread
    _get_step_buffer().clear()

    app = build_workflow()

    input_summary = json.dumps(
        {k: v for k, v in initial_state.items() if k != "enhancement_log"},
        default=str,
    )

    pipeline_error: Optional[Exception] = None
    pipeline_start = time.time_ns()
    try:
        final_state = app.invoke(initial_state)
    except Exception as exc:
        pipeline_error = exc
        final_state = dict(initial_state)
        final_state["status"] = f"PIPELINE_ERROR: {type(exc).__name__}: {exc}"
    pipeline_duration = time.time_ns() - pipeline_start

    output_summary = json.dumps(
        {k: v for k, v in final_state.items() if k != "enhancement_log"},
        default=str,
    )

    # Replay the entire workflow into Galileo atomically
    wf_status = 500 if pipeline_error else 200
    _replay_workflow_to_galileo(
        input_summary, output_summary, pipeline_duration,
        workflow_status_code=wf_status,
    )

    if not defer_upload:
        flush_galileo()

    if pipeline_error and raise_on_error:
        raise pipeline_error

    return final_state
