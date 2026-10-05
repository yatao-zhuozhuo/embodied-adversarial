"""Native benchmark success, independent of dense rewards or agent claims."""
from numbers import Real


def scalar_success(value):
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, RuntimeError):
            return None
    if isinstance(value, bool):
        return value
    if isinstance(value, Real) and value in (0, 1):
        return bool(value)
    return None


def trusted_native_success(info, execution_id=""):
    if not isinstance(info, dict) or info.get("environment_receipt_trusted") is not True:
        return None
    receipt = info.get("environment_receipt")
    if not isinstance(receipt, dict) or receipt.get("schema_version") != "openeta.environment_receipt.v1":
        return None
    if execution_id and receipt.get("execution_id") != execution_id:
        return None
    native = receipt.get("native_success")
    if not isinstance(native, dict) or native.get("source") != "environment.info.success":
        return None
    return scalar_success(native.get("ever_success", native.get("success")))


def native_episode_success(episode):
    """Accept serialized episode records only; missing evidence is failure."""
    execution_id = str((episode.get("metadata") or {}).get("execution_id") or "")
    return any(
        trusted_native_success((step.get("step_result") or {}).get("info"), execution_id) is True
        for step in episode.get("steps", []) if isinstance(step, dict)
    )
