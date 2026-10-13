"""Pydantic v2 API Data Transfer Objects with strict structural privacy validation."""

from pydantic import BaseModel, Field


class ConsentRequest(BaseModel):
    pseudonymous_id: str = Field(min_length=8, max_length=64)
    share_aggregate: bool = False
    share_escalation: bool = False
    policy_version: str = "1.0"


class AggregateCheckinRequest(BaseModel):
    """Contains zero free-text fields."""

    model_config = {"protected_namespaces": ()}

    pseudonymous_id: str = Field(min_length=8, max_length=64)
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    risk_category: str = Field(pattern=r"^(LOW_RISK_TREND|ELEVATED_TREND|ESCALATE_TO_HUMAN)$")
    data_quality: float = Field(ge=0.0, le=1.0)
    model_version: str = "1.0.0"


class EscalationRequest(BaseModel):
    pseudonymous_id: str = Field(min_length=8, max_length=64)
    risk_category: str
    interval_lower: float
    interval_upper: float
    consent_note: bool = False
    optional_note: str | None = Field(default=None, max_length=500)


class EscalationStatusUpdate(BaseModel):
    status: str = Field(pattern=r"^(new|acknowledged|referred|closed)$")
    counsellor_username: str


class LoadEstimateResponse(BaseModel):
    n_students: int
    n_counsellors: int
    capacity_per_day: int
    abstention_rate_used: float
    expected_daily_escalations: float
    capacity_sufficient: bool
