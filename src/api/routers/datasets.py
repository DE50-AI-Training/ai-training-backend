import csv
import io
import os
import shutil

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from sqlmodel import select

from api.database import Dataset, DatasetColumn, SessionDep
from api.schemas.dataset import DatasetCreate, DatasetRead, DatasetTransformation
from config import settings

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
        raise HTTPException(status_code=400, detail="The file must be a CSV file")

    raw_content = await file.read()
    content = raw_content.decode("utf-8")

    sniffer = csv.Sniffer()
    delimiter = sniffer.sniff(content).delimiter
    df = pd.read_csv(io.StringIO(content), delimiter=delimiter)
    
    classes = {}
    for col in df.columns:
        classes[col] = sorted(df[col].astype(str).unique().tolist())

    row_count = df.shape[0]

    dataset = Dataset(
        name=file.filename,
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

    columns = [
        DatasetColumn(
            name=col,
            type="categorical" if df[col].dtype == "object" else "numeric",
            unique_values=df[col].nunique(),
            classes=classes[col],
            null_count=int(df[col].isnull().sum()),
            dataset_id=dataset.id,
        )
        for col in df.columns
    ]

    # Insert all columns into the database in one go
    session.add_all(columns)
    session.commit()
    session.refresh(dataset)

    # On crée le dossier de stockage s'il n'existe pas
    os.makedirs(f"{settings.storage_path}/datasets/{dataset.id}", exist_ok=True)
    with open(f"{settings.storage_path}/datasets/{dataset.id}/dataset.csv", "wb") as f:
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


@router.post("/{dataset_id}/duplicate")
async def duplicate_dataset(dataset_id: int, session: SessionDep) -> DatasetRead:
    db_dataset = session.get(Dataset, dataset_id)
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Create a new dataset with the same properties
    new_dataset = Dataset(
        name=f"{db_dataset.name} (copy)",
        row_count=db_dataset.row_count,
        created_at=pd.Timestamp.now(),
        dataset_type=db_dataset.dataset_type,
        original_file_name=db_dataset.original_file_name,
        delimiter=db_dataset.delimiter,
        is_draft=True,
    )

    session.add(new_dataset)
    session.commit()
    session.refresh(new_dataset)

    # Duplicate the columns
    for column in db_dataset.columns:
        new_column = DatasetColumn(
            name=column.name,
            type=column.type,
            unique_values=column.unique_values,
            null_count=column.null_count,
            dataset_id=new_dataset.id,
        )
        session.add(new_column)

    session.commit()

    # Copy the dataset file to the new location
    source_path = f"{settings.storage_path}/datasets/{db_dataset.id}/dataset.csv"
    destination_path = f"{settings.storage_path}/datasets/{new_dataset.id}/dataset.csv"
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.copy(source_path, destination_path)

    return new_dataset


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: int, session: SessionDep) -> None:
    db_dataset = session.get(Dataset, dataset_id)
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Delete the dataset from the filesystem
    dataset_path = f"{settings.storage_path}/datasets/{db_dataset.id}"
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)

    # TODO: Delete the model files

    session.delete(db_dataset)
    session.commit()
