"""Model adapters for audio QA inference.

Each adapter implements the AbstractModel interface and is auto-registered
via the @register_model decorator. Import modules to trigger registration.
"""

from src.models.base import AbstractModel, ModelLoadError, ModelInferenceError
from src.models import utils as model_utils

# Import adapters to trigger @register_model registration
from src.models import qwen_omni      # noqa: F401
from src.models import qwen2_audio    # noqa: F401
from src.models import moss_audio     # noqa: F401
from src.models import kimi_audio     # noqa: F401
