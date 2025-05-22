import json

def load_model_params_st(path: str):
    # safetensor metadata is so dumb, it only allows you to save string: string maps (even though the storage format is json)
    # and the the library won't let you access the metadata it loads (even though the save function gives you the option to save it)
    # so here we are, opening the model file again, reading the metadata manually and decoding the json string that is stored in a json string...
    with open(path, "rb") as f:
        header = f.read(8)
        n = int.from_bytes(header, "little")
        metadata_bytes = f.read(n)
    metadata = json.loads(metadata_bytes.decode("utf-8")).get("__metadata__", {})
    if "params" in metadata:
        return json.loads(metadata["params"])
    return None
