from typing import Optional

from pydantic import BaseModel


class Checks(BaseModel):
    vertex_ai: Optional[bool] = None
    vector_search: Optional[bool] = None
    firestore: Optional[bool] = None


class HealthCheck(BaseModel):
    status: str
    service: str
    version: str


class ReadyCheck(BaseModel):
    status: str
    checks: Checks
