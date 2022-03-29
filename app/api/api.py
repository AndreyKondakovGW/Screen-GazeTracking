import importlib
from fastapi import APIRouter
from pathlib import Path

from app.api.routes import (
    stream,
)

router = APIRouter()

routes_path = Path(".").absolute()
routes_path = routes_path / Path("app", "api", "routes")

routes_module = importlib.import_module("app.api.routes")

for filename in routes_path.iterdir():
    if filename.name.startswith("__"):
        continue

    router.include_router(routes_module.__getattribute__(filename.name[:-3]).router)

"""
    router.include_router(stream.router)
"""
