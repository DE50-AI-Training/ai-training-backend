- Install and launch Redis (e.g. `docker pull redis && docker run -d --restart unless-stopped -p 6379:6379 redis`)
- Install poetry
- Run `poetry install --with ml,dev`
- Create and edit `.env` file from `.env.example`
- Run the project `poetry run start-dev`
- Run celery workers from the poetry venv `celery -A trainer.trainer.app worker -l INFO`
- See documentation: `localhost:8000`

For windows users:
- If you dont use docker, install redis for windows: https://github.com/redis-windows/redis-windows
- Run celery from a administrator terminal with `--pool=solo` option (only for development)