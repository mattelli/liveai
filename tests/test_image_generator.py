"""Offline tests for src.image_generator: lazy imports, defaults, pipelines."""

import importlib
import sys
import types

import pytest
from PIL import Image

import src.image_generator as ig


# ------------------------------------------------------------------ lazy imports


def test_module_import_is_lazy(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "diffusers", raising=False)
    monkeypatch.delitem(sys.modules, "src.image_generator", raising=False)
    importlib.import_module("src.image_generator")
    assert "torch" not in sys.modules and "diffusers" not in sys.modules


def test_device_override_needs_no_torch(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    gen = ig.ImageGenerator(device="cpu")
    assert gen.device == "cpu"
    assert "torch" not in sys.modules


# ------------------------------------------------------------------ fakes


class FakeCuda:
    def __init__(self, available):
        self._available = available
        self.cache_cleared = False

    def is_available(self):
        return self._available

    def empty_cache(self):
        self.cache_cleared = True


def _install_fake_torch(monkeypatch, cuda_available):
    import contextlib

    torch = types.ModuleType("torch")

    @contextlib.contextmanager
    def inference_mode():
        yield

    torch.cuda = FakeCuda(cuda_available)
    torch.float16, torch.float32 = "float16", "float32"
    torch.manual_seed = lambda seed: ("generator", seed)
    torch.inference_mode = inference_mode
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


class FakePipeline:
    """Records construction and behaves like a diffusers pipeline."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = kwargs.get("model")
        self.images = [object()]
        self.call_args = None

    def to(self, device):
        self.device = device
        return self

    def set_progress_bar_config(self, **kwargs):
        self.progress_bar = kwargs

    def __call__(self, *args, **kwargs):
        self.call_args = (args, kwargs)
        return self


@pytest.fixture
def torch_cpu(monkeypatch):
    return _install_fake_torch(monkeypatch, cuda_available=False)


@pytest.fixture
def torch_cuda(monkeypatch):
    return _install_fake_torch(monkeypatch, cuda_available=True)


@pytest.fixture
def fake_diffusers(monkeypatch):
    created = []

    def register(kind):
        @staticmethod
        def from_pretrained(model, **kwargs):
            pipe = FakePipeline(model=model, **kwargs)
            created.append((kind, pipe))
            return pipe
        return from_pretrained

    diffusers = types.ModuleType("diffusers")
    diffusers.DiffusionPipeline = types.SimpleNamespace(from_pretrained=register("t2i"))
    diffusers.StableDiffusionImg2ImgPipeline = types.SimpleNamespace(from_pretrained=register("i2i"))
    diffusers.StableDiffusionInpaintPipeline = types.SimpleNamespace(from_pretrained=register("inpaint"))
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    return created


# ------------------------------------------------------------------ device / dtype


def test_device_auto_cpu(torch_cpu):
    gen = ig.ImageGenerator()
    assert gen.device == "cpu" and gen._torch_dtype == "float32"


def test_device_auto_cuda(torch_cuda):
    gen = ig.ImageGenerator()
    assert gen.device == "cuda" and gen._torch_dtype == "float16"


def test_defaults_and_model_property():
    gen = ig.ImageGenerator(t2i_model="fake/model", device="cpu", width=640)
    assert gen.t2i_model == "fake/model"
    assert gen.defaults["device"] == "cpu" and gen.defaults["width"] == 640
    assert gen.defaults["seed"] == ig.DEFAULT_SEED


# ------------------------------------------------------------------ check_size


def test_check_size_accepts_multiples_of_8():
    assert ig.check_size(512, "width") == 512


@pytest.mark.parametrize("value", [0, -8, 500, None])
def test_check_size_rejects(value):
    with pytest.raises(ValueError, match="multiple of 8"):
        ig.check_size(value, "width")


# ------------------------------------------------------------------ pipeline management


def test_load_t2i_pipe_is_idempotent(torch_cpu, fake_diffusers):
    gen = ig.ImageGenerator()
    assert gen.is_t2i_pipe_loaded() is False
    assert gen.load_t2i_pipe() is True
    assert gen.load_t2i_pipe() is True  # second call reuses the loaded pipeline
    assert len(fake_diffusers) == 1
    kind, pipe = fake_diffusers[0]
    assert kind == "t2i"
    assert pipe.model == ig.DEFAULT_T2I_MODEL and pipe.device == "cpu"
    assert gen.is_t2i_pipe_loaded() is True


def test_load_i2i_pipe_switches_mode(torch_cpu, fake_diffusers):
    gen = ig.ImageGenerator()
    gen.load_i2i_pipe(in_paint=False)
    gen.load_i2i_pipe(in_paint=False)  # same mode: reused
    assert [kind for kind, _ in fake_diffusers] == ["i2i"]
    gen.load_i2i_pipe(in_paint=True)  # mode change: reloads
    assert [kind for kind, _ in fake_diffusers] == ["i2i", "inpaint"]
    assert gen.is_i2i_pipe_loaded() is True


def test_unload_pipes(torch_cuda, fake_diffusers):
    gen = ig.ImageGenerator()
    assert gen.unload_t2i_pipe() is False  # nothing loaded
    gen.load_t2i_pipe()
    assert gen.unload_t2i_pipe() is True
    assert gen.unload_t2i_pipe() is False
    assert torch_cuda.cuda.cache_cleared  # VRAM release was attempted


# ------------------------------------------------------------------ text2image


def test_text2image_requires_pipe():
    with pytest.raises(ValueError, match="not initialized"):
        ig.ImageGenerator().text2image("a cat")


def test_text2image_uses_defaults(torch_cpu, fake_diffusers):
    gen = ig.ImageGenerator()
    gen.load_t2i_pipe()
    gen.text2image("a cat")
    args, kwargs = fake_diffusers[-1][1].call_args
    assert args[0] == "a cat" and args[1] == ig.DEFAULT_NEGATIVE_PROMPT
    assert kwargs["width"] == 512 and kwargs["height"] == 512
    assert kwargs["num_inference_steps"] == 50
    assert gen.last_seed == ig.DEFAULT_SEED


def test_text2image_explicit_params_override(torch_cpu, fake_diffusers):
    gen = ig.ImageGenerator(width=640, steps=20, seed=7, negative_prompt="no cats")
    gen.load_t2i_pipe()
    gen.text2image("a cat", width=256, height=384, steps=5, seed=1, negative_prompt="")
    args, kwargs = fake_diffusers[-1][1].call_args
    assert args[1] == ""  # explicit empty negative prompt is kept
    assert kwargs["width"] == 256 and kwargs["height"] == 384
    assert kwargs["num_inference_steps"] == 5
    assert gen.last_seed == 1


def test_text2image_negative_seed_is_randomized(torch_cpu, fake_diffusers):
    gen = ig.ImageGenerator()
    gen.load_t2i_pipe()
    gen.text2image("a cat", seed=-1)
    assert 0 <= gen.last_seed < 2**32


def test_text2image_rejects_bad_size(torch_cpu, fake_diffusers):
    gen = ig.ImageGenerator()
    gen.load_t2i_pipe()
    with pytest.raises(ValueError, match="multiple of 8"):
        gen.text2image("a cat", width=500)


# ------------------------------------------------------------------ image2image


def _png(tmp_path, name="in.png"):
    img = Image.new("RGB", (8, 8), "red")
    path = tmp_path / name
    img.save(path)
    return path


def test_image2image_without_mask_needs_no_mask(torch_cpu, fake_diffusers, tmp_path):
    gen = ig.ImageGenerator()
    gen.load_i2i_pipe(in_paint=False)
    out = gen.image2image(_png(tmp_path), "make it blue")
    kind, pipe = fake_diffusers[-1]
    assert kind == "i2i" and out in pipe.images
    args, kwargs = pipe.call_args
    assert "mask_image" not in kwargs  # plain img2img must not require a mask
    assert kwargs["strength"] == ig.DEFAULT_DENOISE
    assert kwargs["guidance_scale"] == ig.DEFAULT_GUIDANCE


def test_image2image_inpaint_uses_mask(torch_cpu, fake_diffusers, tmp_path):
    gen = ig.ImageGenerator()
    gen.load_i2i_pipe(in_paint=True)
    with pytest.raises(ValueError, match="mask"):
        gen.image2image(_png(tmp_path), "make it blue")
    mask = _png(tmp_path, "mask.png")
    gen.image2image(_png(tmp_path), "make it blue", mask=mask, denoise=0.8, guidance=9.0)
    _, kwargs = fake_diffusers[-1][1].call_args
    assert "mask_image" in kwargs
    assert kwargs["strength"] == 0.8 and kwargs["guidance_scale"] == 9.0


def test_image2image_missing_input_file(torch_cpu, fake_diffusers, tmp_path):
    gen = ig.ImageGenerator()
    gen.load_i2i_pipe()
    with pytest.raises(FileNotFoundError):
        gen.image2image(tmp_path / "nope.png", "prompt")


def test_image2image_requires_pipe():
    with pytest.raises(ValueError, match="not initialized"):
        ig.ImageGenerator().image2image("in.png", "prompt")


# ------------------------------------------------------------------ save


def test_save_writes_file(tmp_path):
    out = tmp_path / "sub" / "out.png"
    assert ig.ImageGenerator.save(Image.new("RGB", (4, 4)), out) is True
    assert out.exists()


def test_save_rejects_directory(tmp_path):
    with pytest.raises(ValueError, match="directory"):
        ig.ImageGenerator.save(Image.new("RGB", (2, 2)), tmp_path)


def test_save_without_image_returns_false():
    assert ig.ImageGenerator.save(None, "out.png") is False
