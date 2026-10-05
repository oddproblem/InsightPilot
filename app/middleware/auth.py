# EXTENSION POINT: Replace or extend API key validation here.
# To add OAuth: implement token verification in _validate_bearer_token()
# and set the same request.state fields so downstream routes work unchanged.
# To add JWT: decode and verify here, populate request.state.tenant_id from claims.

import json
from typing import cast

from fastapi import Depends, HTTPException, Request, Response
from fastapi.params import Depends as DependsClass
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from app.models.responses import ErrorResponse
from app.services.api_key_service import api_key_service

# Routes that bypass authentication entirely
_PUBLIC_ROUTES: set[tuple[str, str]] = {
    ("GET", "/health"),
    ("GET", "/docs"),
    ("GET", "/openapi.json"),
    ("GET", "/redoc"),
    ("GET", "/"),
    ("GET", "/index.html"),
    ("GET", "/style.css"),
    ("GET", "/app.js"),
    ("GET", "/favicon.ico"),
}

# Routes that require admin role (checked after auth passes)
_ADMIN_ROUTES: set[tuple[str, str]] = {
    ("POST", "/v1/keys"),
}


def _is_admin_route(method: str, path: str) -> bool:
    if (method, path) in _ADMIN_ROUTES:
        return True
    # DELETE /v1/keys/{id}
    if method == "DELETE" and path.startswith("/v1/keys/"):
        return True
    return False


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        method = request.method
        path = request.url.path

        # Skip auth for public routes and static UI assets
        if (method, path) in _PUBLIC_ROUTES or path.startswith(("/static", "/assets")):
            return await call_next(request)

        # Extract Bearer token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content=json.loads(
                    ErrorResponse(
                        code="MISSING_API_KEY",
                        message="Authorization header with Bearer token is required.",
                    ).model_dump_json()
                ),
            )

        token = auth_header[len("Bearer ") :]
        validated = api_key_service.validate_key(token)

        if validated is None:
            return JSONResponse(
                status_code=401,
                content=json.loads(
                    ErrorResponse(
                        code="INVALID_API_KEY",
                        message="The provided API key is invalid or has been revoked.",
                    ).model_dump_json()
                ),
            )

        # Attach auth info to request state for downstream use
        request.state.api_key = validated
        request.state.tenant_id = validated.tenant_id

        # Admin-only routes
        if _is_admin_route(method, path) and validated.role != "admin":
            return JSONResponse(
                status_code=403,
                content=json.loads(
                    ErrorResponse(
                        code="INSUFFICIENT_PERMISSIONS",
                        message="This endpoint requires admin role.",
                    ).model_dump_json()
                ),
            )

        return await call_next(request)


def require_role(minimum_role: str = "user") -> DependsClass:
    """FastAPI dependency factory that enforces a minimum role on an endpoint.

    Usage:
        @router.post("/...", dependencies=[Depends(require_role("admin"))])

    Roles in ascending permission order: user < admin
    """
    _role_rank = {"user": 0, "admin": 1}

    async def _check(request: Request) -> None:
        api_key = getattr(request.state, "api_key", None)
        if api_key is None:
            raise HTTPException(
                status_code=401,
                detail={"code": "UNAUTHENTICATED", "message": "Authentication required."},
            )
        key_role = getattr(api_key, "role", "user")
        if _role_rank.get(key_role, 0) < _role_rank.get(minimum_role, 0):
            raise HTTPException(
                status_code=403,
                detail={
                    "code": "INSUFFICIENT_PERMISSIONS",
                    "message": f"This endpoint requires '{minimum_role}' role.",
                },
            )

    return cast(DependsClass, Depends(_check))
