import logging
import os
import shutil
import tempfile

import datasets
import lakefs_sdk
from fastapi import APIRouter, File, UploadFile, Depends
from fastapi.security import HTTPBasicCredentials, HTTPBasic

from surogate import huggingface
from surogate.jobs.lakefs_utils import ensure_repository
from surogate.utils import write_dataset_info

logging.basicConfig(level=logging.ERROR)

lakefs_endpoint = os.getenv("LAKEFS_ENDPOINT", "https://lakefs.densemax.local/api/v1")

router = APIRouter(tags=["dataset"])


@router.get(path="/dataset/download_size")
def get_download_size(dataset_id: str, allow_patterns: list = []):
    try:
        download_size_in_bytes = huggingface.get_huggingface_download_size(dataset_id, allow_patterns,
                                                                           repo_type="dataset")
    except Exception as e:
        logging.error(f"Error in get_model_download_size: {type(e).__name__}: {e}")
        return {"status": "error", "message": "An internal error has occurred."}

    return {"status": "success", "data": download_size_in_bytes}


@router.post(path="/dataset/upload")
async def create_repo_from_file(repo_id: str, branch: str = "main", file: UploadFile = File(...),
                           credentials: HTTPBasicCredentials = Depends(HTTPBasic())):
    username = credentials.username
    password = credentials.password
    tmp_path = None

    lakefs_configuration = lakefs_sdk.Configuration(host=lakefs_endpoint,
                                                    username=username,
                                                    password=password)
    lakefs_configuration.verify_ssl = False

    with lakefs_sdk.ApiClient(lakefs_configuration) as api_client:
        ensure_repository(repo_id, "main", file.filename, api_client, "dataset")

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = os.path.join(tmpdir, file.filename)
                with open(tmp_path, "wb") as dst:
                    shutil.copyfileobj(file.file, dst)

                dataset = datasets.load_dataset(tmpdir)
                dataset.save_to_disk(f"lakefs://{repo_id}/{branch}",
                                     storage_options={"username": username, "password": password,
                                                      "host": lakefs_endpoint, "verify_ssl": False})
                write_dataset_info(dataset, path=tmpdir)
                ds_info_file = os.path.join(tmpdir, "surogate_info.json")

                try:
                    objectsApi = lakefs_sdk.ObjectsApi(api_client)
                    objectsApi.upload_object(
                        repository=repo_id,
                        branch=branch,
                        path="surogate_info.json",
                        content=ds_info_file
                    )

                    commitsApi = lakefs_sdk.CommitsApi(api_client)
                    commitsApi.commit(
                        repository=repo_id,
                        branch=branch,
                        commit_creation=lakefs_sdk.CommitCreation(
                            message="Add dataset " + file.filename
                        )
                    )
                except lakefs_sdk.ApiException as e:
                    if not (e.body and "message" in e.body and "no changes" in e.body["message"].lower()):
                        raise e

        except Exception as e:
            logging.error(f"Error in load_dataset_from_file: {type(e).__name__}: {e}")
            return {"status": "error", "message": "An internal error has occurred."}
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    return {"status": "success"}
