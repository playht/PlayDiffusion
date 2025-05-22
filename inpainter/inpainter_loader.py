def load_inpainter(type: str, **kwargs):
    if type == 'maskgct_inpainter':
        from .masklm_text import load_maskgct_inpainter
        return load_maskgct_inpainter(**kwargs)
    else:
        raise ValueError(f"Could not create inpainter '{type}' with {kwargs}")
