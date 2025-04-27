from fastapi import APIRouter

from api.schemas.model import ModelWithArchitecture

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/")
async def get_models() -> list[ModelWithArchitecture]:
    return [
        {
            "id": 1,
            "name": "Test Model",
            "architecture": {
                "id": 1,
                "input_size": 784,
                "output_size": 10,
                "activation": "relu",
                "layers": [128, 64],
            },
        },
        {
            "id": 2,
            "name": "Another Model",
            "architecture": {
                "id": 2,
                "input_size": 784,
                "output_size": 10,
                "activation": "sigmoid",
                "layers": [256, 128],
            },
        },
    ]
