import logging

# Import each task defensively. Some tasks need optional heavy deps that are not
# required for the demo/inference (e.g. `jailbreak` needs `fastchat`). A missing
# optional dep simply omits that task from `task_mapper` instead of breaking the
# import for everything else.
_logger = logging.getLogger("task")

task_mapper = {}


def _register(name, module, cls):
    try:
        mod = __import__(f"tasks.{module}", fromlist=[cls])
        task_mapper[name] = getattr(mod, cls)
    except Exception as exc:  # noqa: BLE001 - optional dependency may be missing
        _logger.warning("Task '%s' unavailable (%s): %s", name, module, exc)


_register("paradetox", "paradetox", "ParaDetoxProbInferenceForStyle")
_register("shakespeare", "shakespeare", "ShakespeareProbInferenceForStyle")
_register("formality", "formality", "FormalityProbInferenceForStyle")
_register("sentiment", "sentiment", "SentimentProbInferenceForStyle")
_register("format", "format", "FormatProbInferenceForStyle")
_register("emotive", "emotive", "EmotiveProbInferenceForStyle")
_register("jailbreak", "jailbreak", "JailBreakProbInferenceForStyle")
_register("demo", "demo", "DemoProbInferenceForStyle")


def load_task(name):
    if name not in task_mapper:
        raise ValueError(f"Unrecognized or unavailable dataset `{name}`")
    return task_mapper[name]
