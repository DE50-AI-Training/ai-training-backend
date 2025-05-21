import redis as r

from api.schemas.training import TrainingRead, TrainingStatusEnum
from config import settings

redis = r.Redis.from_url(settings.redis_url, decode_responses=True, db=0)


def get_training(model_id: int) -> TrainingRead | None:
    training = redis.get(f"training:{model_id}")
    if not training:
        return None
    training_data = TrainingRead.model_validate_json(training)
    return training_data


def set_training(training: TrainingRead) -> None:
    redis.set(
        f"training:{training.model_id}",
        TrainingRead.model_dump_json(training),
    )


def update_training_status(model_id: int, status: TrainingStatusEnum) -> None:
    training_data = get_training(model_id)
    training_data.status = status
    set_training(training_data)

def trainer_stop(model_id: int) -> None:
    redis.set(f"stop_signal:{model_id}", "True", ex=3600)

def trainer_should_stop(model_id: int) -> bool:
    stop_signal = redis.get(f"stop_signal:{model_id}")
    if stop_signal:
        redis.delete(f"stop_signal:{model_id}")
        return True
    return False
