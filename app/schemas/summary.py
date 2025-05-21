from pydantic import BaseModel


class URLRequest(BaseModel):
    url: str


class StatusResponse(BaseModel):
    status: str
    result: dict | None = None
    error: str | None = None
