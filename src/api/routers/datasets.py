import io
from os import makedirs

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from sqlmodel import select

from api.database import Dataset, SessionDep
from api.schemas.dataset import DatasetCreate, DatasetRead, DatasetTransformation
from config import settings
import csv

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("/")
async def get_datasets(session: SessionDep) -> list[DatasetRead]:
    statement = select(Dataset).where(Dataset.is_draft == False)
    datasets = session.exec(statement).all()
    return datasets


@router.post("/")
async def upload_dataset(
    session: SessionDep, file: UploadFile = File(...)
) -> DatasetRead:
    if not file.content_type.startswith("text/csv"):
        raise ValueError("The file must be a CSV file")

    raw_content = await file.read()
    content = raw_content.decode("utf-8")

    sniffer = csv.Sniffer()
    delimiter = sniffer.sniff(content).delimiter
    df = pd.read_csv(io.StringIO(content), delimiter=delimiter)
    columns = df.columns.tolist()
    row_count = df.shape[0]
    unique_values_per_column=[df[col].nunique() for col in columns]
    
    dataset = Dataset(
        name=file.filename,
        columns=columns,
        unique_values_per_column=unique_values_per_column,
        row_count=row_count,
        created_at=pd.Timestamp.now(),
        dataset_type="csv",
        original_file_name=file.filename,
        delimiter=delimiter,
        is_draft=True,
    )

    session.add(dataset)
    session.commit()
    session.refresh(dataset)

    # On crée le dossier de stockage s'il n'existe pas
    makedirs(f"{settings.storage_path}/datasets", exist_ok=True)
    with open(f"{settings.storage_path}/datasets/{dataset.id}.csv", "wb") as f:
        f.write(raw_content)

    return dataset


@router.put("/{dataset_id}")
async def transform_dataset(
    dataset_id: int, transformation: DatasetTransformation
) -> DatasetRead:
    # On va chercher le dataset, on effectue la transformation et on le remplace dans le fs
    pass


@router.post("/{dataset_id}")
async def create_dataset(
    dataset_id: int, dataset: DatasetCreate, session: SessionDep
) -> DatasetRead:

    db_dataset = session.get(Dataset, dataset_id)
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    db_dataset.name = dataset.name
    db_dataset.is_draft = False

    session.add(db_dataset)
    session.commit()
    session.refresh(db_dataset)

    return db_dataset
