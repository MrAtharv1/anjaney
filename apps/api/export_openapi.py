"""Exports openapi.json to enable typed TypeScript client generation for Phase 10."""

import json
from pathlib import Path

from api.main import app

out_path = Path("apps/api/openapi.json")
out_path.parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(app.openapi(), f, indent=2)

print(f"OpenAPI schema successfully written to {out_path}")
