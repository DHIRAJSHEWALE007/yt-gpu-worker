"""worker/engines: Inference backend factory and registry."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type
import yaml

from worker.engines.base import Engine, InsufficientVRAMError
from worker.engines.fake import FakeEngine
from worker.engines.flux2_klein import Flux2KleinEngine
from worker.engines.sdxl_lightning import SDXLLightningEngine
from worker.engines.wan22_ti2v5b import Wan22TI2V5BEngine
from worker.engines.zimage_turbo import ZImageTurboEngine

logger = logging.getLogger(__name__)

_ENGINES_REGISTRY: Dict[str, Type[Engine]] = {
    "fake": FakeEngine,
    "sdxl_lightning": SDXLLightningEngine,
    "zimage_turbo": ZImageTurboEngine,
    "flux2_klein": Flux2KleinEngine,
    "wan21_t2v_1_3b": Wan22TI2V5BEngine,
    "wan22_ti2v5b": Wan22TI2V5BEngine,  # Backwards compatibility alias
}


def load_models_config() -> Dict[str, Any]:
    """Load models registry from worker/models.yaml (single source of truth)."""
    p_worker = Path(__file__).resolve().parent.parent / "models.yaml"
    if p_worker.is_file():
        try:
            data = yaml.safe_load(p_worker.read_text(encoding="utf-8"))
            return data.get("models", {}) if isinstance(data, dict) else {}
        except Exception as e:
            logger.warning("Could not read worker/models.yaml: %s", e)
    p_root = Path(__file__).resolve().parent.parent.parent / "models.yaml"
    if p_root.is_file():
        try:
            data = yaml.safe_load(p_root.read_text(encoding="utf-8"))
            return data.get("models", {}) if isinstance(data, dict) else {}
        except Exception as e:
            logger.warning("Could not read root models.yaml: %s", e)
    return {}


def get_engine(engine_id: str, config: Optional[Dict[str, Any]] = None) -> Engine:
    """Instantiate an engine by identifier using models.yaml configuration."""
    cls = _ENGINES_REGISTRY.get(engine_id)
    if not cls:
        raise ValueError(
            f"Unknown engine '{engine_id}'. Available engines: {list(_ENGINES_REGISTRY.keys())}"
        )

    if config is None:
        models = load_models_config()
        cfg = models.get(engine_id)
        if not cfg and engine_id == "wan22_ti2v5b":
            cfg = models.get("wan21_t2v_1_3b", {})
        elif not cfg and engine_id == "wan21_t2v_1_3b":
            cfg = models.get("wan22_ti2v5b", {})
        config = cfg or {}

    return cls(config=config)
