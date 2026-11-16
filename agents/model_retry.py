"""Handle a provider rejecting a model-generated tool call before execution."""

from langchain.agents.middleware import ModelRetryMiddleware


def tool_generation_failed(error: Exception) -> bool:
    if getattr(error, "status_code", None) != 400:
        return False
    body = getattr(error, "body", None)
    details = body.get("error", body) if isinstance(body, dict) else None
    return isinstance(details, dict) and details.get("code") == "tool_use_failed"


def retry_tool_generation() -> ModelRetryMiddleware:
    return ModelRetryMiddleware(
        max_retries=2, retry_on=tool_generation_failed, on_failure="error", initial_delay=0.25,
    )
