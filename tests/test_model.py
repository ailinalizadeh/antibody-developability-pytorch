import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from antibody_developability.model import AntibodyDevelopabilityCNN


def test_model_output_shape():
    model = AntibodyDevelopabilityCNN()
    heavy = torch.randint(0, 22, (4, 128))
    light = torch.randint(0, 22, (4, 128))
    logits = model(heavy, light)

    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()
