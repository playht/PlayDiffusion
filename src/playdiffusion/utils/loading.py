import boto3
import os
import tqdm
import zipfile
from decouple import config

from urllib import request


class S3Progress:
    def __init__(self, name, total_size):
        self.total_size = total_size
        self.progress = tqdm.tqdm(
            total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc=name
        )

    def __call__(self, downloaded):
        self.progress.update(downloaded)
        if downloaded >= self.total_size:
            self.progress.close()


class RequestProgress:
    def __init__(self, name):
        self.progress = None
        self.name = name

    def __call__(self, block_num, block_size, total_size):
        if self.progress is None:
            self.progress = tqdm.tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=self.name,
            )
        downloaded = block_num * block_size
        self.progress.update(downloaded - self.progress.n)
        if downloaded >= total_size:
            self.progress.close()


def save_resource(url: str, path: str, verbose=True, recursive=False):
    """Retrieve a resource from a URL or local path and save it to a local path."""
    if verbose:
        print(f"Downloading {url} to {path}")

    file_name = url.split("/")[-1]
    if url.endswith(".zip"):
        os.makedirs(path, exist_ok=True)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if url.startswith("s3://"):
        s3 = boto3.resource("s3", region_name='us-east-2',
                            aws_access_key_id=config('AWS_ACCESS_KEY_ID', None),
                            aws_secret_access_key=config('AWS_SECRET_ACCESS_KEY', None))
        bucket_name, key = url[5:].split("/", 1)
        bucket = s3.Bucket(bucket_name)

        if recursive and os.path.isdir(path):
            # Handle directory download (only level 1 files)
            if not key.endswith('/'):
                key = key + '/'

            os.makedirs(path, exist_ok=True)

            # List and download only level 1 files
            for obj_summary in bucket.objects.filter(Prefix=key):
                if obj_summary.key == key:
                    continue

                relative_path = obj_summary.key[len(key):]
                if not relative_path or '/' in relative_path:
                    continue

                local_path = os.path.join(path, relative_path)
                obj = s3.Object(bucket_name, obj_summary.key)
                if verbose:
                    progress = S3Progress(os.path.basename(local_path), obj.content_length)
                else:
                    progress = None

                obj.download_file(local_path, Callback=progress)
        else:
            # Handle single file download
            obj = s3.Object(bucket_name, key)
            if verbose:
                progress = S3Progress(os.path.basename(path), obj.content_length)
            else:
                progress = None

            if url.endswith(".zip"):
                obj.download_file(os.path.join(path, file_name), Callback=progress)
            else:
                obj.download_file(path, Callback=progress)

    elif url.startswith("https://") or url.startswith("http://"):
        if verbose:
            progress = RequestProgress(os.path.basename(path))
        else:
            progress = None
        request.urlretrieve(url, path, progress)
    else:
        raise ValueError(f"Unknown URL: {url}")

    if url.endswith(".zip") and not recursive:
        with zipfile.ZipFile(os.path.join(path, file_name)) as f:
            f.extractall(path)

        os.remove(os.path.join(path, file_name))
