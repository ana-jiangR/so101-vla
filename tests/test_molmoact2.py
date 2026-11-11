"""MolmoAct2 wrapper without a GPU or the 22 GB checkpoint: config loading and layer selection."""

import pytest

from so101_vla.policies.molmoact2 import MolmoAct2Config, load_molmoact2_config


def test_yaml_config_loads_and_overrides():
    cfg = load_molmoact2_config("configs/policy/molmoact2_so101.yaml", quantization="nf4")
    assert cfg.quantization == "nf4" and cfg.image_size == (640, 480)
    assert cfg.camera_map == {"cam0": "front", "cam1": "main/wrist_cam"}
    assert cfg.revision and cfg.weights_revision  # both repos pinned


def test_config_rejects_bad_quantization():
    with pytest.raises(ValueError):
        MolmoAct2Config(quantization="int4")
    with pytest.raises(ValueError):
        MolmoAct2Config(device="cpu", quantization="int8")
    MolmoAct2Config(device="cpu", quantization="none")


def test_quantizes_only_selected_linears():
    torch = pytest.importorskip("torch")
    bnb = pytest.importorskip("bitsandbytes")
    from torch import nn

    from so101_vla.policies.molmoact2 import DEFAULT_QUANTIZE_PREFIXES, _quantize_linears

    def block():
        return nn.ModuleDict({"att_proj": nn.Linear(8, 24, bias=False), "ff_out": nn.Linear(8, 8)})

    model = nn.Module()
    model.lm_head = nn.Linear(8, 16, bias=False)
    model.model = nn.Module()
    model.model.transformer = nn.Module()
    model.model.transformer.blocks = nn.ModuleList([block(), block()])
    model.model.transformer.wte = nn.Embedding(16, 8)
    model.model.vision_backbone = nn.Module()
    model.model.vision_backbone.image_vit = nn.Module()
    model.model.vision_backbone.image_vit.patch_embedding = nn.Linear(4, 8)
    model.model.vision_backbone.image_vit.wq = nn.Linear(8, 8)
    model.model.vision_backbone.image_projector = nn.Linear(8, 8)
    model.model.action_expert = nn.Module()
    model.model.action_expert.qkv = nn.Linear(8, 24)

    with torch.no_grad():
        bias = model.model.transformer.blocks[0].ff_out.bias.clone()
    info = _quantize_linears(model, DEFAULT_QUANTIZE_PREFIXES, "int8")
    assert info["layers"] == 5  # 2 blocks x 2, ViT wq
    tb = model.model.transformer.blocks
    assert isinstance(tb[0].att_proj, bnb.nn.Linear8bitLt) and type(model.lm_head) is nn.Linear
    assert isinstance(model.model.vision_backbone.image_vit.wq, bnb.nn.Linear8bitLt)
    assert torch.equal(tb[0].ff_out.bias, bias)
    for kept in (model.model.vision_backbone.image_vit.patch_embedding, model.model.vision_backbone.image_projector,
                 model.model.action_expert.qkv):
        assert type(kept) is nn.Linear
