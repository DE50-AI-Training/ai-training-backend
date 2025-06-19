from fastapi import APIRouter, File, UploadFile

from api.database import SessionDep
from api.schemas.dataset import DatasetCreate, DatasetRead, DatasetTransformation
from api.services.dataset_service import DatasetService

router = APIRouter(
    prefix="/datasets",
    tags=["Datasets"],
)


@router.get(
    "/",
    summary="Retrieve all datasets",
    description="Returns the list of all finalized datasets (non-drafts) available in the system.",
    response_description="List of datasets with their metadata and columns",
)
async def get_datasets(session: SessionDep) -> list[DatasetRead]:
    """
    Retrieves all finalized datasets.

    - **Automatically filters** draft datasets
    - **Includes metadata**: row count, type, creation date
    - **Includes columns**: name, type, unique values, possible classes
    """
    return DatasetService.get_all_datasets(session)


@router.post(
    "/",
    summary="Upload a new dataset",
    description="Upload a CSV file and create a dataset in draft mode. The file is automatically analyzed to detect column types and metadata.",
    response_description="Dataset created in draft mode with all its metadata",
    status_code=201,
)
async def upload_dataset(
    session: SessionDep,
    file: UploadFile = File(
        ...,
        description="CSV file to upload. Must have .csv extension and text/csv content-type",
    ),
) -> DatasetRead:
    """
    Upload and analyze a CSV file to create a new dataset.

    **Automatic process:**
    - Automatic CSV delimiter detection
    - Column type analysis (numeric/categorical)
    - Statistics calculation (unique values, null values)
    - Class extraction for categorical variables
    - Secure file storage

    **The dataset is created in draft mode** and must be finalized with POST /{dataset_id}

    **Supported formats:** CSV with delimiters: comma, semicolon, tab
    """
    return await DatasetService.upload_dataset(session, file)


@router.put(
    "/{dataset_id}",
    summary="Transform a dataset",
    description="Apply transformations to an existing dataset (feature under development).",
    response_description="Transformed dataset",
)
async def transform_dataset(
    dataset_id: int, transformation: DatasetTransformation
) -> DatasetRead:
    """
    Apply transformations to an existing dataset.

    **🚧 Feature under development**

    **Planned transformations:**
    - Numeric data normalization
    - Categorical variable encoding
    - Missing value handling
    - Row/column filtering
    """
    return DatasetService.transform_dataset(dataset_id, transformation)


@router.post(
    "/{dataset_id}",
    summary="Finalize a draft dataset",
    description="Finalizes a dataset in draft mode by giving it a definitive name and making it usable for training.",
    response_description="Finalized dataset ready for use",
)
async def create_dataset(
    dataset_id: int, dataset: DatasetCreate, session: SessionDep
) -> DatasetRead:
    """
    Finalizes a draft dataset to make it usable.

    **Actions performed:**
    - Assignment of definitive name
    - Status change from draft to finalized
    - Dataset becomes available for model creation

    **Prerequisites:** Dataset must exist and be in draft mode
    """
    return DatasetService.finalize_dataset(dataset_id, dataset, session)


@router.post(
    "/{dataset_id}/duplicate",
    summary="Duplicate a dataset",
    description="Creates a complete copy of an existing dataset, including its data and metadata.",
    response_description="New dataset (copy) created in draft mode",
    status_code=201,
)
async def duplicate_dataset(dataset_id: int, session: SessionDep) -> DatasetRead:
    """
    Duplicates an existing dataset with all its data.

    **Copied elements:**
    - Original CSV file
    - All metadata (columns, types, statistics)
    - Structure and configuration

    **Name is automatically suffixed** with " (copy)"
    **Copy is created in draft mode** and can be renamed
    **Use cases:** Experimentation, backup, multiple versions
    """
    return DatasetService.duplicate_dataset(dataset_id, session)


@router.delete(
    "/{dataset_id}",
    summary="Delete a dataset",
    description="Permanently deletes a dataset and all its associated files. This action is irreversible.",
    response_description="Deletion confirmed",
    status_code=204,
)
async def delete_dataset(dataset_id: int, session: SessionDep) -> None:
    """
    Permanently deletes a dataset and its data.

    **⚠️ WARNING: Irreversible action!**

    **Deleted elements:**
    - Dataset CSV file
    - Database metadata
    - Complete storage folder
    - All associated columns

    **TODO:** Delete models using this dataset

    **Prerequisites:** Dataset must not be used by active models
    """
    DatasetService.delete_dataset(dataset_id, session)
