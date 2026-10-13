"""Privacy structure audits, consent gating tests, k-anonymity suppression, and erasure validation."""

import pytest
from api.main import app
from api.models.database import Base, engine
from api.schemas import dto
from httpx import ASGITransport, AsyncClient


@pytest.fixture(autouse=True)
async def setup_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield


@pytest.mark.asyncio
async def test_schema_privacy_invariant_zero_free_text():
    """STRUCTURAL PRIVACY AUDIT: Walk all DTO schemas to verify NO free-text fields exist.

    (Except the explicitly permitted optional student consent note).
    """
    banned_names = {"text", "content", "message", "raw_utterance", "transcript"}

    for name in dir(dto):
        cls = getattr(dto, name)
        if isinstance(cls, type) and issubclass(cls, dto.BaseModel) and cls is not dto.BaseModel:
            for field_name in cls.model_fields:
                assert field_name.lower() not in banned_names, (
                    f"Privacy Invariant Broken: Schema {cls.__name__} exposes free-text field '{field_name}'"
                )


@pytest.mark.asyncio
async def test_consent_gating_and_erasure():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        pid = "STUDENT_XYZ_12345"

        # 1. Reject aggregate logging without prior consent
        res = await ac.post(
            "/api/v1/aggregates",
            json={
                "pseudonymous_id": pid,
                "date": "2026-10-13",
                "risk_category": "LOW_RISK_TREND",
                "data_quality": 0.95,
            },
        )
        assert res.status_code == 403

        # 2. Grant consent
        res = await ac.post(
            "/api/v1/consent",
            json={"pseudonymous_id": pid, "share_aggregate": True, "share_escalation": True},
        )
        assert res.status_code == 201

        # 3. Aggregate logging now accepted
        res = await ac.post(
            "/api/v1/aggregates",
            json={
                "pseudonymous_id": pid,
                "date": "2026-10-13",
                "risk_category": "LOW_RISK_TREND",
                "data_quality": 0.95,
            },
        )
        assert res.status_code == 202

        # 4. Right to erasure (DPDP Act)
        res = await ac.delete(f"/api/v1/me/{pid}")
        assert res.status_code == 200

        # 5. Subsequent attempts without consent fail again
        res = await ac.post(
            "/api/v1/aggregates",
            json={
                "pseudonymous_id": pid,
                "date": "2026-10-13",
                "risk_category": "LOW_RISK_TREND",
                "data_quality": 0.95,
            },
        )
        assert res.status_code == 403


@pytest.mark.asyncio
async def test_k_anonymity_suppression():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Seed 3 distinct students with ELEVATED_TREND (k < 5)
        for i in range(3):
            pid = f"STUDENT_{i:04d}"
            await ac.post(
                "/api/v1/consent",
                json={"pseudonymous_id": pid, "share_aggregate": True, "share_escalation": False},
            )
            await ac.post(
                "/api/v1/aggregates",
                json={
                    "pseudonymous_id": pid,
                    "date": "2026-10-13",
                    "risk_category": "ELEVATED_TREND",
                    "data_quality": 1.0,
                },
            )

        # Check summary endpoint
        res = await ac.get("/api/v1/dashboard/summary")
        assert res.status_code == 200
        data = res.json()["summary"]

        # Cell with 3 students must be suppressed
        assert data["ELEVATED_TREND"]["suppressed"] is True
        assert "< 5" in str(data["ELEVATED_TREND"]["count"])
