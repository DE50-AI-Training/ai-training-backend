from fastapi import APIRouter, File, UploadFile
from sqlmodel import select

from api.database import Dataset, SessionDep
from api.schemas.dataset import DatasetCreate, DatasetRead, DatasetTransformation

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("/")
async def get_datasets(session: SessionDep) -> list[DatasetRead]:
    statement = select(Dataset)
    datasets = session.exec(statement).all()
    return datasets


@router.post("/")
async def upload_dataset(file: UploadFile = File(...)) -> DatasetRead:
    # Ici on insert les données du dataset dans la bdd pour avoir un id, on spécifie que c'est un brouillon avec un field
    # On load le fichier dans la mémoire avec Redis grace à DataPreparation (on map le dataset avec son id)
    pass


@router.put("/{dataset_id}")
async def transform_dataset(
    dataset_id: int, transformation: DatasetTransformation
) -> DatasetRead:
    # On va chercher le dataset dans Redis avec l'id
    # On effectue la transformation
    # On remplace le dataset dans Redis
    pass


@router.post("/{dataset_id}")
async def create_dataset(dataset_id: int, dataset: DatasetCreate) -> DatasetRead:
    # On va chercher le dataset transformé dans Redis avec l'id
    # On edit le dataset de la bdd avec les données du dataset (nom, nb de lignes, etc)
    # On le sauvegarde de manière permanente dans le fs
    # On le supprime de Redis
    # On retourne le dataset
    pass
