"""Typed loading of the platform's YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

CONFIG_DIR = Path(__file__).resolve().parent
MODEL_POLICY_PATH = CONFIG_DIR / "model_policy.yaml"
VERIFICATION_PROFILES_PATH = CONFIG_DIR / "verification_profiles.yaml"


class _Strict(BaseModel):
    """Reject unknown keys, so a typo in a policy file fails loudly."""

    model_config = ConfigDict(extra="forbid")


class ModelSpec(_Strict):
    id: str
    num_ctx: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    think: bool | None = None

    @model_validator(mode="after")
    def _output_fits_context(self) -> "ModelSpec":
        if self.max_output_tokens >= self.num_ctx:
            raise ValueError(f"{self.id}: max_output_tokens must be smaller than num_ctx")
        return self


class RolePolicy(_Strict):
    strategy: Literal["fixed", "alternate_of_developer"] = "fixed"
    primary: str | None = None
    fallback: str | None = None


class RetryPolicy(_Strict):
    # The specification sets two attempts per model as a hard maximum.
    attempts_per_model: int = Field(default=2, ge=1, le=2)


class TimeoutPolicy(_Strict):
    inference_seconds: float = Field(default=900, gt=0)


class MemoryPolicy(_Strict):
    unload_between_models: bool = True
    keep_alive: str | int = "5m"
    unload_wait_seconds: float = Field(default=20, ge=0)


class ContextPolicy(_Strict):
    chars_per_token: float = Field(default=3.0, gt=0)
    reserved_tokens: int = Field(default=1400, ge=0)
    max_section_chars: int = Field(default=6000, gt=0)
    max_conversation_turns: int = Field(default=12, gt=0)
    retrieval_top_k: int = Field(default=4, ge=0)
    retrieval_file_chars: int = Field(default=2400, gt=0)


class AgentPolicy(_Strict):
    max_llm_calls_per_task: int = Field(default=12, gt=0)


class WorkflowPolicy(_Strict):
    discovery_max_rounds: int = Field(default=5, gt=0)
    max_repair_cycles: int = Field(default=2, ge=0)


class ModelPolicy(_Strict):
    models: dict[str, ModelSpec]
    roles: dict[str, RolePolicy]
    retry: RetryPolicy = RetryPolicy()
    timeouts: TimeoutPolicy = TimeoutPolicy()
    memory: MemoryPolicy = MemoryPolicy()
    context: ContextPolicy = ContextPolicy()
    agent: AgentPolicy = AgentPolicy()
    workflow: WorkflowPolicy = WorkflowPolicy()

    @model_validator(mode="after")
    def _roles_reference_known_models(self) -> "ModelPolicy":
        for role, policy in self.roles.items():
            for alias in (policy.primary, policy.fallback):
                if alias is not None and alias not in self.models:
                    raise ValueError(f"role {role!r} references unknown model {alias!r}")
            if policy.strategy == "fixed" and policy.primary is None:
                raise ValueError(f"role {role!r} needs a primary model")
        return self

    def spec_for(self, model_id: str) -> ModelSpec:
        for spec in self.models.values():
            if spec.id == model_id:
                return spec
        raise KeyError(f"Model {model_id!r} is not in the model policy")

    def _alternate_of(self, model_id: str) -> str | None:
        for spec in self.models.values():
            if spec.id != model_id:
                return spec.id
        return None

    def chain_for(self, role: str, developer_model: str | None = None) -> list[str]:
        """Return the ordered models for *role*: preferred first, then its fallback."""
        policy = self.roles[role]
        if policy.strategy == "alternate_of_developer":
            developer = developer_model or self.chain_for("developer")[0]
            alternate = self._alternate_of(developer)
            chain = [alternate, developer] if alternate else [developer]
        else:
            chain = [self.models[policy.primary].id]  # type: ignore[index]
            if policy.fallback:
                chain.append(self.models[policy.fallback].id)
        return list(dict.fromkeys(chain))


class VerificationProfile(_Strict):
    profile_id: str = ""
    description: str
    executable: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    args: list[str]
    result_parser: Literal["pytest", "unittest", "tap"]
    test_file_patterns: list[str] = Field(min_length=1)
    helper_file_patterns: list[str] = Field(default_factory=list)
    support_files: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=120, gt=0)
    authoring_notes: str = ""


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def load_model_policy(path: Path | str | None = None) -> ModelPolicy:
    return ModelPolicy.model_validate(_read_yaml(Path(path or MODEL_POLICY_PATH)))


def load_verification_profiles(
    extra_path: Path | str | None = None,
) -> dict[str, VerificationProfile]:
    """Load the allow-list, plus a product's own human-maintained additions."""
    raw = dict(_read_yaml(VERIFICATION_PROFILES_PATH).get("profiles", {}))
    if extra_path and Path(extra_path).is_file():
        raw.update(_read_yaml(Path(extra_path)).get("profiles", {}))
    return {
        profile_id: VerificationProfile.model_validate({**body, "profile_id": profile_id})
        for profile_id, body in raw.items()
    }
