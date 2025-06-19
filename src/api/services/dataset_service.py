import csv
import io
import os
import shutil
from typing import List

import pandas as pd
from fastapi import HTTPException, UploadFile
from sqlmodel import Session, select

from api.database import Dataset, DatasetColumn
from api.schemas.dataset import DatasetCreate, DatasetRead, DatasetTransformation
from config import settings


class DatasetService:
    """
    Service layer for dataset management.
    
    Handles dataset operations:
    - CSV file upload and analysis
    - Dataset finalization and transformation
    - File storage and cleanup
    - Dataset duplication and deletion
    """

    @staticmethod
    def get_all_datasets(session: Session) -> List[DatasetRead]:
        """Get all finalized datasets"""
        statement = select(Dataset).where(Dataset.is_draft == False)
        datasets = session.exec(statement).all()
        return datasets

    @staticmethod
    async def upload_dataset(session: Session, file: UploadFile) -> DatasetRead:
        """Upload and analyze a CSV file to create a new dataset"""
        if not file.content_type.startswith("text/csv"):
            raise HTTPException(status_code=400, detail="The file must be a CSV file")

        raw_content = await file.read()
        content = raw_content.decode("utf-8")

        # Automatic CSV format detection
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(content).delimiter
        df = pd.read_csv(io.StringIO(content), delimiter=delimiter)

        # Extract unique values for each column (for categorical analysis)
        classes = {}
        for col in df.columns:
            classes[col] = sorted(df[col].astype(str).unique().tolist())

        row_count = df.shape[0]

        # Create dataset in draft mode
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

        # Analyze and create column metadata
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

        # Bulk insert for performance
        session.add_all(columns)
        session.commit()
        session.refresh(dataset)

        # Store file to persistent storage
        os.makedirs(f"{settings.storage_path}/datasets/{dataset.id}", exist_ok=True)
        with open(
            f"{settings.storage_path}/datasets/{dataset.id}/dataset.csv", "wb"
        ) as f:
            f.write(raw_content)

        return dataset

    @staticmethod
    def transform_dataset(
        dataset_id: int, transformation: DatasetTransformation
    ) -> DatasetRead:
        """Apply transformations to an existing dataset"""
        # TODO: Implement transformation logic
        pass

    @staticmethod
    def finalize_dataset(
        dataset_id: int, dataset: DatasetCreate, session: Session
    ) -> DatasetRead:
        """Finalize a draft dataset"""
        db_dataset = session.get(Dataset, dataset_id)
        if not db_dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        db_dataset.name = dataset.name
        db_dataset.is_draft = False

        session.add(db_dataset)
        session.commit()
        session.refresh(db_dataset)

        return db_dataset

    @staticmethod
    def duplicate_dataset(dataset_id: int, session: Session) -> DatasetRead:
        """Duplicate an existing dataset"""
        db_dataset = session.get(Dataset, dataset_id)
        if not db_dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Create new dataset with copied metadata
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

        # Duplicate column definitions
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

        # Copy physical file to new location
        source_path = f"{settings.storage_path}/datasets/{db_dataset.id}/dataset.csv"
        destination_path = (
            f"{settings.storage_path}/datasets/{new_dataset.id}/dataset.csv"
        )
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.copy(source_path, destination_path)

        return new_dataset

    @staticmethod
    def delete_dataset(dataset_id: int, session: Session) -> None:
        """Delete a dataset and its files"""
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
