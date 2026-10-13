"""Database models enforcing the Zero-Free-Text and Audit Trail privacy invariants."""

from datetime import datetime

from api.models.database import Base
from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column


class ConsentRecord(Base):
    __tablename__ = "consents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pseudonymous_id: Mapped[str] = mapped_column(
        String(64), index=True, unique=True, nullable=False
    )
    share_aggregate: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    share_escalation: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    policy_version: Mapped[str] = mapped_column(String(16), default="1.0", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class AggregateCheckin(Base):
    """Opt-in anonymous metric logging. Structurally contains ZERO free text."""

    __tablename__ = "aggregates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pseudonymous_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    checkin_date: Mapped[str] = mapped_column(String(10), index=True, nullable=False)  # YYYY-MM-DD
    risk_category: Mapped[str] = mapped_column(String(32), nullable=False)
    data_quality_score: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    model_version: Mapped[str] = mapped_column(String(16), default="1.0.0", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class EscalationQueue(Base):
    """Counselor triage queue. Student note is consent-gated."""

    __tablename__ = "escalations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pseudonymous_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    risk_category: Mapped[str] = mapped_column(String(32), nullable=False)
    interval_lower: Mapped[float] = mapped_column(Float, nullable=False)
    interval_upper: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="new", nullable=False)
    assigned_counsellor: Mapped[str | None] = mapped_column(String(64), nullable=True)
    optional_consent_note: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    counsellor_username: Mapped[str] = mapped_column(String(64), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    target_id: Mapped[str] = mapped_column(String(64), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
