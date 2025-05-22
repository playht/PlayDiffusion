from play_inpainter.utils.get_resource import get_resource

def get_all_checkpoints_from_preset(preset: dict, download=True, models_dir=None):
    for key in preset:
        if key.endswith('checkpoint') or (key.endswith('file') and preset[key].startswith('s3://')):
            preset[key] = get_resource(preset[key], download, models_dir)
        elif isinstance(preset[key], dict):
            get_all_checkpoints_from_preset(preset[key], download, models_dir)

    return preset
