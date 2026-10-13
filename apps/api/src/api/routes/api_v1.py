"""API v1 Endpoints: Bundle delivery, consent gating, k-anonymity dashboards, and erasure."""

import hashlib
import json

from api.config import settings
from api.models.database import get_db
from api.models.entities import AggregateCheckin, AuditLog, ConsentRecord, EscalationQueue
from api.schemas.dto import (
    AggregateCheckinRequest,
    ConsentRequest,
    EscalationRequest,
    EscalationStatusUpdate,
    LoadEstimateResponse,
)
from fastapi import APIRouter, Depends, Header, HTTPException, Response, status
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/api/v1")


# 1. Model Bundle Delivery
@router.get("/model/bundle")
async def get_model_bundle(if_none_match: str | None = Header(default=None)):
    if not settings.bundle_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model bundle not built. Run 'anjaney export bundle' first.",
        )

    content = settings.bundle_path.read_bytes()
    etag = f'"{hashlib.sha256(content).hexdigest()}"'

    if if_none_match == etag:
        return Response(status_code=status.HTTP_304_NOT_MODIFIED)

    data = json.loads(content.decode("utf-8"))
    return {"bundle": data, "etag": etag}


@router.get("/model/version")
async def get_model_version():
    return {"model_version": settings.app_version, "encoder": "all-MiniLM-L6-v2-int8"}


# 2. Granular Consent Management
@router.post("/consent", status_code=status.HTTP_201_CREATED)
async def record_consent(req: ConsentRequest, db: AsyncSession = Depends(get_db)):
    stmt = select(ConsentRecord).where(ConsentRecord.pseudonymous_id == req.pseudonymous_id)
    res = await db.execute(stmt)
    record = res.scalar_one_or_none()

    if record:
        record.share_aggregate = req.share_aggregate
        record.share_escalation = req.share_escalation
        record.policy_version = req.policy_version
    else:
        record = ConsentRecord(
            pseudonymous_id=req.pseudonymous_id,
            share_aggregate=req.share_aggregate,
            share_escalation=req.share_escalation,
            policy_version=req.policy_version,
        )
        db.add(record)

    await db.commit()
    return {"status": "recorded", "pseudonymous_id": req.pseudonymous_id}


# 3. Anonymized Aggregate Check-ins (Zero Free-Text)
@router.post("/aggregates", status_code=status.HTTP_202_ACCEPTED)
async def record_aggregate(req: AggregateCheckinRequest, db: AsyncSession = Depends(get_db)):
    # Verify opt-in consent exists
    stmt = select(ConsentRecord).where(
        ConsentRecord.pseudonymous_id == req.pseudonymous_id,
        ConsentRecord.share_aggregate.is_(True),
    )
    res = await db.execute(stmt)
    if not res.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No consent recorded for aggregate sharing.",
        )

    entry = AggregateCheckin(
        pseudonymous_id=req.pseudonymous_id,
        checkin_date=req.date,
        risk_category=req.risk_category,
        data_quality_score=req.data_quality,
        model_version=req.model_version,
    )
    db.add(entry)
    await db.commit()
    return {"status": "accepted"}


# 4. Consent-Gated Escalations
@router.post("/escalations", status_code=status.HTTP_201_CREATED)
async def submit_escalation(req: EscalationRequest, db: AsyncSession = Depends(get_db)):
    stmt = select(ConsentRecord).where(
        ConsentRecord.pseudonymous_id == req.pseudonymous_id,
        ConsentRecord.share_escalation.is_(True),
    )
    res = await db.execute(stmt)
    if not res.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Student has not consented to escalation sharing.",
        )

    note = req.optional_note if req.consent_note else None
    esc = EscalationQueue(
        pseudonymous_id=req.pseudonymous_id,
        risk_category=req.risk_category,
        interval_lower=req.interval_lower,
        interval_upper=req.interval_upper,
        optional_consent_note=note,
    )
    db.add(esc)
    await db.commit()
    return {"status": "escalated", "id": esc.id}


# 5. Wellness Cell Dashboard Summary (k-Anonymity Suppression)
@router.get("/dashboard/summary")
async def get_dashboard_summary(db: AsyncSession = Depends(get_db)):
    stmt = select(
        AggregateCheckin.risk_category,
        func.count(AggregateCheckin.id).label("total_checkins"),
        func.count(func.distinct(AggregateCheckin.pseudonymous_id)).label("distinct_students"),
    ).group_by(AggregateCheckin.risk_category)

    res = await db.execute(stmt)
    rows = res.all()

    summary: dict[str, dict] = {}
    for cat, total, distinct in rows:
        # k-anonymity suppression rule
        if distinct < settings.k_anonymity_min:
            summary[cat] = {
                "count": "< 5 (Suppressed for Privacy)",
                "suppressed": True,
            }
        else:
            summary[cat] = {
                "count": total,
                "distinct_students": distinct,
                "suppressed": False,
            }

    return {"k_min": settings.k_anonymity_min, "summary": summary}


# 6. Counselor Escalation Queue
@router.get("/dashboard/escalations")
async def list_escalations(db: AsyncSession = Depends(get_db)):
    stmt = select(EscalationQueue).order_by(EscalationQueue.created_at.desc())
    res = await db.execute(stmt)
    items = res.scalars().all()
    return [
        {
            "id": it.id,
            "pseudonymous_id": it.pseudonymous_id,
            "risk_category": it.risk_category,
            "interval": [it.interval_lower, it.interval_upper],
            "status": it.status,
            "note": it.optional_consent_note,
            "created_at": it.created_at.isoformat(),
        }
        for it in items
    ]


@router.patch("/dashboard/escalations/{escalation_id}")
async def update_escalation_status(
    escalation_id: int, req: EscalationStatusUpdate, db: AsyncSession = Depends(get_db)
):
    stmt = select(EscalationQueue).where(EscalationQueue.id == escalation_id)
    res = await db.execute(stmt)
    esc = res.scalar_one_or_none()
    if not esc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Escalation not found")

    esc.status = req.status
    esc.assigned_counsellor = req.counsellor_username

    audit = AuditLog(
        counsellor_username=req.counsellor_username,
        action=f"update_status_{req.status}",
        target_id=str(escalation_id),
    )
    db.add(audit)
    await db.commit()
    return {"status": "updated", "escalation_id": escalation_id}


# 7. Counselor Capacity Load Estimator
@router.get("/dashboard/load-estimate", response_model=LoadEstimateResponse)
async def get_load_estimate(
    n_students: int = 2000,
    n_counsellors: int = 3,
    capacity_per_day: int = 5,
):
    # Dynamically read abstention rate from latest conformal calibration run
    conformal_runs = sorted(settings.experiments_path.glob("conformal/*/conformal_results.json"))
    abstention_rate = 0.15  # Default baseline estimate

    if conformal_runs:
        try:
            with open(conformal_runs[-1], encoding="utf-8") as f:
                c_data = json.load(f)
            abstention_rate = (
                c_data.get("operating_point_sweep", {})
                .get("optimal_operating_point", {})
                .get("abstention_rate", 0.15)
            )
        except Exception:
            pass

    expected_daily = n_students * abstention_rate * 0.05
    total_capacity = n_counsellors * capacity_per_day

    return LoadEstimateResponse(
        n_students=n_students,
        n_counsellors=n_counsellors,
        capacity_per_day=capacity_per_day,
        abstention_rate_used=round(abstention_rate, 3),
        expected_daily_escalations=round(expected_daily, 1),
        capacity_sufficient=expected_daily <= total_capacity,
    )


# 8. Right to Erasure (DPDP Act 2023)
@router.delete("/me/{pseudonymous_id}", status_code=status.HTTP_200_OK)
async def erase_student_records(pseudonymous_id: str, db: AsyncSession = Depends(get_db)):
    await db.execute(delete(ConsentRecord).where(ConsentRecord.pseudonymous_id == pseudonymous_id))
    await db.execute(
        delete(AggregateCheckin).where(AggregateCheckin.pseudonymous_id == pseudonymous_id)
    )
    await db.execute(
        delete(EscalationQueue).where(EscalationQueue.pseudonymous_id == pseudonymous_id)
    )
    await db.commit()
    return {"status": "erased", "pseudonymous_id": pseudonymous_id}
