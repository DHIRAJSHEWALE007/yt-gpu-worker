"""worker/engines: Inference backend factory and registry."""

from __future__ import annotations

from typing import Dict, Type

from worker.engines.base import Engine, InsufficientVRAMError
from worker.engines.fake import FakeEngine
from worker.engines.flux2_klein import Flux2KleinEngine
from worker.engines.sdxl_lightning import SDXLLightningEngine
from worker.engines.wan22_ti2v5b import Wan22TI2V5BEngine
from worker.engines.zimage_turbo import ZImageTurboEngine

_ENGINES_REGISTRY: Dict[str, Type[Engine]] = {
    "fake": FakeEngine,
    "sdxl_lightning": SDXLLightningEngine,
    "zimage_turbo": ZImageTurboEngine,
    "flux2_klein": Flux2KleinEngine,
    "wan22_ti2v5b": Wan22TI2V5BEngine,
}


def get_engine(engine_id: str, config: dict | None = None) -> Engine:
    """Instantiate an engine by identifier."""
    cls = _ENGINES_REGISTRY.get(engine_id)
    if not cls:
        raise ValueError(
            f"Unknown engine '{engine_id}'. Available engines: {list(_ENGINES_REGISTRY.keys())}"
        )
    return cls(config=config)
