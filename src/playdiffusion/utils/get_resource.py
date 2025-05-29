import os
import re
from playdiffusion.utils.loading import save_resource

_USER_DOWNLOAD_DIR = os.path.join(os.path.expanduser("~"), ".cache", "playht", "resources")
RESOURCE_DOWNLOAD_DIR = os.environ.get("PLAYHT_CACHE_DIR", _USER_DOWNLOAD_DIR)

def get_resource(uri: str, download = True, models_dir = None):
    if uri.startswith('file://'):
        path = uri[7:]
    elif uri.startswith('file:'):
        path = uri[5:]
    elif uri.startswith('env:'):
        path = os.environ[uri[4:]]
    elif uri.startswith('s3://'):
        models_dir = models_dir or RESOURCE_DOWNLOAD_DIR
        path = os.path.join(models_dir, "s3", uri[5:])
        if download and not os.path.exists(path):
            save_resource(uri, path, recursive=True)
    else:
        # search for something that looks like a scheme and if not found, treat the uri as a path
        m = re.match(r"^\w+\:", uri)
        # also check length of match, if it's 2, we treat it as a windows absolute path
        if m is None or m.span()[1] == 2:
            path = uri
        else:
            raise ValueError(f"Unsupported uri scheme in '{uri}'")


    if os.path.isfile(path):
        return path
    if os.path.exists(path):
        print(f"Target path '{path}' for uri '{uri}' exists and is a directory, this must be for SGLang HF loading")
        return path
    raise ValueError(f"Target path '{path}' for uri '{uri}' does not exist")
