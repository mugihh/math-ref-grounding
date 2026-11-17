"""Path helpers shared across image-reading stages."""

from pathlib import Path


def resolve_image_path(image_path: str | Path, images_root: str | Path | None) -> Path:
    """Resolve a logical (possibly relative) image_path against an optional root.

    Artifacts store ``image_path`` as a root-free logical id (e.g.
    ``en/topic/video1/000.jpg``). Stages that actually open the file pass the
    machine-local ``images_root`` to locate it; absolute paths are returned as-is.
    """
    path = Path(image_path)
    if path.is_absolute() or images_root is None:
        return path
    return Path(images_root) / path
