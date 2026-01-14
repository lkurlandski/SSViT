"""
Tests.
"""

import math
from typing import Optional
from typing import Sequence

from torch import nn
import pytest

from src.architectures import Identity
from src.architectures import PatchEncoder
from src.architectures import ConvPatchEncoder
from src.architectures import HierarchicalConvPatchEncoder
from src.architectures import PatchEncoderLowMem
from src.architectures import PatchEncoderLowMemSwitchMoE
from src.architectures import PatchPositionalityEncoder
from src.architectures import SinusoidalPositionalEncoding
from src.architectures import LearnedPositionalEncoding
from src.architectures import FiLM
from src.architectures import FiLMNoP
from src.architectures import MalConv
from src.architectures import MalConvLowMem
from src.architectures import MalConvGCG
from src.architectures import Classifier
from src.architectures import MalConvClassifier
from src.architectures import ViTClassifier
from src.architectures import HierarchicalClassifier
from src.architectures import HierarchicalMalConvClassifier
from src.architectures import HierarchicalViTClassifier
from src.architectures import StructuralClassifier
from src.architectures import StructuralMalConvClassifier
from src.architectures import StructuralViTClassifier
from src.binanal import HierarchicalLevel
from src.binanal import HierarchicalStructure
from src.binanal import HierarchicalStructureNone
from src.binanal import HierarchicalStructureCoarse
from src.helpers import Design
from src.helpers import Architecture
from src.helpers import PatcherArchitecture
from src.helpers import PositionalEncodingArchitecture
from src.helpers import PatchPositionalEncodingArchitecture
from src.main import get_model


class TestGetModel:

    @pytest.mark.parametrize("arch", [Architecture.MCV, Architecture.MC2, Architecture.MCG])
    @pytest.mark.parametrize("num_guides", [0, 3, 7])
    def test_flat_malconv(self, arch: Architecture, num_guides: int) -> None:
        model = get_model(Design.FLAT, arch, num_guides=num_guides)

        assert isinstance(model, MalConvClassifier)

        if arch == Architecture.MCV:
            assert isinstance(model.backbone, MalConv)
        elif arch == Architecture.MC2:
            assert isinstance(model.backbone, MalConvLowMem)
        elif arch == Architecture.MCG:
            assert isinstance(model.backbone, MalConvGCG)
        else:
            raise NotImplementedError()

        if num_guides == 0:
            assert isinstance(model.filmer, FiLMNoP)
        else:
            assert isinstance(model.filmer, FiLM)
            assert model.filmer.guide_dim == num_guides

    @pytest.mark.parametrize("parch", PatcherArchitecture)
    @pytest.mark.parametrize("posenc", PositionalEncodingArchitecture)
    @pytest.mark.parametrize("patchposenc", PatchPositionalEncodingArchitecture)
    @pytest.mark.parametrize("num_guides", [0, 3, 7])
    @pytest.mark.parametrize("max_length", [None, 2 ** 20, 2 ** 22])
    def test_flat_transformer(self, parch: PatcherArchitecture, posenc: PositionalEncodingArchitecture, patchposenc: PatchPositionalEncodingArchitecture, num_guides: int, max_length: Optional[int]) -> None:

        def _get_model() -> ViTClassifier:
            return get_model(
                design=Design.FLAT,
                arch=Architecture.VIT,
                parch=parch,
                posenc=posenc,
                patchposenc=patchposenc,
                num_guides=num_guides,
                max_length=max_length,
            )

        if patchposenc == PatchPositionalEncodingArchitecture.ABS:
            with pytest.raises(NotImplementedError):
                _get_model()
            return

        if patchposenc in (PatchPositionalEncodingArchitecture.BTH, PatchPositionalEncodingArchitecture.ABS) and max_length is None:
            with pytest.raises(ValueError):
                _get_model()
            return

        if parch in (PatcherArchitecture.CNV, PatcherArchitecture.HCV) and posenc == PositionalEncodingArchitecture.LEARNED and max_length is None:
            with pytest.raises(ValueError):
                _get_model()
            return

        model =  _get_model()
        assert isinstance(model, ViTClassifier)

        if parch == PatcherArchitecture.BAS:
            assert isinstance(model.patcher, PatchEncoder)
        elif parch == PatcherArchitecture.CNV:
            assert isinstance(model.patcher, ConvPatchEncoder)
        elif parch == PatcherArchitecture.HCV:
            assert isinstance(model.patcher, HierarchicalConvPatchEncoder)
        elif parch == PatcherArchitecture.MEM:
            assert isinstance(model.patcher, PatchEncoderLowMem)
        elif parch == PatcherArchitecture.EXP:
            assert isinstance(model.patcher, PatchEncoderLowMemSwitchMoE)
        else:
            raise NotImplementedError()

        if posenc == PositionalEncodingArchitecture.NONE:
            assert isinstance(model.backbone.posencoder, nn.Identity)
        elif posenc == PositionalEncodingArchitecture.FIXED:
            assert isinstance(model.backbone.posencoder, SinusoidalPositionalEncoding)
        elif posenc == PositionalEncodingArchitecture.LEARNED:
            assert isinstance(model.backbone.posencoder, LearnedPositionalEncoding)
            if parch in (PatcherArchitecture.CNV, PatcherArchitecture.HCV):
                assert max_length is not None
                assert model.backbone.posencoder.max_len == math.ceil(max_length / 4096) + 1
            else:
                assert model.backbone.posencoder.max_len == 256 + 1
        else:
            raise NotImplementedError()

        if patchposenc == PatchPositionalEncodingArchitecture.NONE:
            assert isinstance(model.patchposencoder, Identity)
        elif patchposenc == PatchPositionalEncodingArchitecture.REL:
            assert isinstance(model.patchposencoder, PatchPositionalityEncoder)
            assert model.patchposencoder.max_length is None
        elif patchposenc == PatchPositionalEncodingArchitecture.BTH:
            assert isinstance(model.patchposencoder, PatchPositionalityEncoder)
            assert model.patchposencoder.max_length is not None
        elif patchposenc == PatchPositionalEncodingArchitecture.ABS:
            assert isinstance(model.patchposencoder, PatchPositionalityEncoder)
            assert model.patchposencoder.max_length is not None
        else:
            raise NotImplementedError()

        if num_guides == 0:
            assert isinstance(model.filmer, FiLMNoP)
        else:
            assert isinstance(model.filmer, FiLM)
            assert model.filmer.guide_dim == num_guides

    @pytest.mark.parametrize("arch", [Architecture.MCV, Architecture.MC2, Architecture.MCG])
    @pytest.mark.parametrize("num_guides", [0, 3, 7])
    @pytest.mark.parametrize("structures", [[], list(HierarchicalStructureNone), list(HierarchicalStructureCoarse)])
    @pytest.mark.parametrize("share_embeddings", [False, True])
    def test_hierarchical_malconv(self, arch: Architecture, num_guides: int, structures: Sequence[HierarchicalStructure], share_embeddings: bool) -> None:

        def _get_model() -> HierarchicalMalConvClassifier:
            return get_model(
                design=Design.HIERARCHICAL,
                arch=arch,
                num_guides=num_guides,
                structures=structures,
                share_embeddings=share_embeddings,
            )

        if len(structures) == 0:
            with pytest.raises(ValueError):
                _get_model()
            return

        model = _get_model()
        assert isinstance(model, HierarchicalMalConvClassifier)
        assert len(structures) == len(model.embeddings) == len(model.filmers) == len(model.backbones)

        if arch == Architecture.MCV:
            assert all(isinstance(backbone, MalConv) for backbone in model.backbones)
        elif arch == Architecture.MC2:
            assert all(isinstance(backbone, MalConvLowMem) for backbone in model.backbones)
        elif arch == Architecture.MCG:
            assert all(isinstance(backbone, MalConvGCG) for backbone in model.backbones)
        else:
            raise NotImplementedError()

        if num_guides == 0:
            assert all(isinstance(filmer, FiLMNoP) for filmer in model.filmers)
        else:
            assert all(isinstance(filmer, FiLM) for filmer in model.filmers)
            assert all(filmer.guide_dim == num_guides for filmer in model.filmers)

        ids_embd = set(id(module) for module in model.embeddings)
        ids_film = set(id(module) for module in model.filmers)
        if share_embeddings:
            assert len(ids_embd) == 1
            assert len(ids_film) == 1
        else:
            assert len(ids_embd) == len(structures)
            assert len(ids_film) == len(structures)

    @pytest.mark.parametrize("parch", PatcherArchitecture)
    @pytest.mark.parametrize("posenc", PositionalEncodingArchitecture)
    @pytest.mark.parametrize("patchposenc", PatchPositionalEncodingArchitecture)
    @pytest.mark.parametrize("num_guides", [0, 3, 7])
    @pytest.mark.parametrize("max_length", [None, 2 ** 20, 2 ** 22])
    @pytest.mark.parametrize("structures", [[], list(HierarchicalStructureNone), list(HierarchicalStructureCoarse)])
    @pytest.mark.parametrize("share_embeddings", [False, True])
    def test_hierarchical_transformer(self, parch: PatcherArchitecture, posenc: PositionalEncodingArchitecture, patchposenc: PatchPositionalEncodingArchitecture, num_guides: int, max_length: Optional[int], structures: list[HierarchicalStructure], share_embeddings: bool) -> None:

        def _get_model() -> HierarchicalViTClassifier:
            return get_model(
                design=Design.HIERARCHICAL,
                arch=Architecture.VIT,
                parch=parch,
                posenc=posenc,
                patchposenc=patchposenc,
                num_guides=num_guides,
                structures=structures,
                max_length=max_length,
                share_embeddings=share_embeddings,
            )

        if len(structures) == 0:
            with pytest.raises(ValueError):
                _get_model()
            return

        if patchposenc == PatchPositionalEncodingArchitecture.ABS:
            with pytest.raises(NotImplementedError):
                _get_model()
            return

        if patchposenc in (PatchPositionalEncodingArchitecture.BTH, PatchPositionalEncodingArchitecture.ABS) and max_length is None:
            with pytest.raises(ValueError):
                _get_model()
            return

        if parch in (PatcherArchitecture.CNV, PatcherArchitecture.HCV) and posenc == PositionalEncodingArchitecture.LEARNED and max_length is None:
            with pytest.raises(ValueError):
                _get_model()
            return

        if parch in (PatcherArchitecture.CNV, PatcherArchitecture.HCV):
            with pytest.raises(ValueError):
                _get_model()
            return

        model = _get_model()
        assert isinstance(model, HierarchicalViTClassifier)
        assert len(structures) == len(model.embeddings) == len(model.filmers) == len(model.patchers) == len(model.norms) == len(model.patchposencoders)

        if parch == PatcherArchitecture.BAS:
            assert all(isinstance(patcher, PatchEncoder) for patcher in model.patchers)
        elif parch == PatcherArchitecture.CNV:
            assert all(isinstance(patcher, ConvPatchEncoder) for patcher in model.patchers)
        elif parch == PatcherArchitecture.HCV:
            assert all(isinstance(patcher, HierarchicalConvPatchEncoder) for patcher in model.patchers)
        elif parch == PatcherArchitecture.MEM:
            assert all(isinstance(patcher, PatchEncoderLowMem) for patcher in model.patchers)
        elif parch == PatcherArchitecture.EXP:
            assert all(isinstance(patcher, PatchEncoderLowMemSwitchMoE) for patcher in model.patchers)
        else:
            raise NotImplementedError()

        if posenc == PositionalEncodingArchitecture.NONE:
            assert isinstance(model.backbone.posencoder, nn.Identity)
        elif posenc == PositionalEncodingArchitecture.FIXED:
            assert isinstance(model.backbone.posencoder, SinusoidalPositionalEncoding)
        elif posenc == PositionalEncodingArchitecture.LEARNED:
            assert isinstance(model.backbone.posencoder, LearnedPositionalEncoding)
            if parch in (PatcherArchitecture.CNV, PatcherArchitecture.HCV):
                assert max_length is not None
                assert model.backbone.posencoder.max_len == math.ceil(max_length / 4096) + 1
            else:
                assert model.backbone.posencoder.max_len == 256 + 1
        else:
            raise NotImplementedError()

        if patchposenc == PatchPositionalEncodingArchitecture.NONE:
            assert all(isinstance(patchposencoder, Identity) for patchposencoder in model.patchposencoders)
        elif patchposenc == PatchPositionalEncodingArchitecture.REL:
            assert all(isinstance(patchposencoder, PatchPositionalityEncoder) for patchposencoder in model.patchposencoders)
            assert all(patchposencoder.max_length is None for patchposencoder in model.patchposencoders)
        elif patchposenc == PatchPositionalEncodingArchitecture.BTH:
            assert all(isinstance(patchposencoder, PatchPositionalityEncoder) for patchposencoder in model.patchposencoders)
            assert all(patchposencoder.max_length is not None for patchposencoder in model.patchposencoders)
        elif patchposenc == PatchPositionalEncodingArchitecture.ABS:
            assert all(isinstance(patchposencoder, PatchPositionalityEncoder) for patchposencoder in model.patchposencoders)
            assert all(patchposencoder.max_length is not None for patchposencoder in model.patchposencoders)
        else:
            raise NotImplementedError()

        if num_guides == 0:
            assert all(isinstance(filmer, FiLMNoP) for filmer in model.filmers)
        else:
            assert all(isinstance(filmer, FiLM) for filmer in model.filmers)
            assert all(filmer.guide_dim == num_guides for filmer in model.filmers)

        ids_embd = set(id(module) for module in model.embeddings)
        ids_film = set(id(module) for module in model.filmers)
        if share_embeddings:
            assert len(ids_embd) == 1
            assert len(ids_film) == 1
        else:
            assert len(ids_embd) == len(structures)
            assert len(ids_film) == len(structures)

    def test_structural_malconv(self, ) -> None:
        design = Design.STRUCTURAL

    @pytest.mark.parametrize("parch", PatcherArchitecture)
    @pytest.mark.parametrize("posenc", PositionalEncodingArchitecture)
    @pytest.mark.parametrize("patchposenc", PatchPositionalEncodingArchitecture)
    @pytest.mark.parametrize("num_guides", [0, 3, 7])
    @pytest.mark.parametrize("max_length", [None, 2 ** 20, 2 ** 22])
    @pytest.mark.parametrize("structures", [[], list(HierarchicalStructureNone), list(HierarchicalStructureCoarse)])
    @pytest.mark.parametrize("share_embeddings", [False, True])
    @pytest.mark.parametrize("share_patchers", [False, True])
    def test_structural_transformer(self, parch: PatcherArchitecture, posenc: PositionalEncodingArchitecture, patchposenc: PatchPositionalEncodingArchitecture, num_guides: int, max_length: Optional[int], structures: list[HierarchicalStructure], share_embeddings: bool, share_patchers: bool) -> None:

        def _get_model() -> StructuralViTClassifier:
            return get_model(
                design=Design.STRUCTURAL,
                arch=Architecture.VIT,
                parch=parch,
                posenc=posenc,
                patchposenc=patchposenc,
                num_guides=num_guides,
                structures=structures,
                max_length=max_length,
                share_embeddings=share_embeddings,
                share_patchers=share_patchers,
            )

        if len(structures) == 0:
            with pytest.raises(ValueError):
                _get_model()
            return

        if patchposenc == PatchPositionalEncodingArchitecture.ABS:
            with pytest.raises(NotImplementedError):
                _get_model()
            return

        if patchposenc in (PatchPositionalEncodingArchitecture.BTH, PatchPositionalEncodingArchitecture.ABS) and max_length is None:
            with pytest.raises(ValueError):
                _get_model()
            return

        if parch in (PatcherArchitecture.CNV, PatcherArchitecture.HCV) and posenc == PositionalEncodingArchitecture.LEARNED and max_length is None:
            with pytest.raises(ValueError):
                _get_model()
            return

        if patchposenc != PatchPositionalEncodingArchitecture.NONE:
            with pytest.raises(NotImplementedError):
                _get_model()
            return

        model = _get_model()
        assert isinstance(model, StructuralViTClassifier)
        assert len(structures) == len(model.embeddings) == len(model.filmers) == len(model.patchers) == len(model.norms) == len(model.patchposencoders)

        if parch == PatcherArchitecture.BAS:
            assert all(isinstance(patcher, PatchEncoder) for patcher in model.patchers)
        elif parch == PatcherArchitecture.CNV:
            assert all(isinstance(patcher, ConvPatchEncoder) for patcher in model.patchers)
        elif parch == PatcherArchitecture.HCV:
            assert all(isinstance(patcher, HierarchicalConvPatchEncoder) for patcher in model.patchers)
        elif parch == PatcherArchitecture.MEM:
            assert all(isinstance(patcher, PatchEncoderLowMem) for patcher in model.patchers)
        elif parch == PatcherArchitecture.EXP:
            assert all(isinstance(patcher, PatchEncoderLowMemSwitchMoE) for patcher in model.patchers)
        else:
            raise NotImplementedError()

        if posenc == PositionalEncodingArchitecture.NONE:
            assert isinstance(model.backbone.posencoder, nn.Identity)
        elif posenc == PositionalEncodingArchitecture.FIXED:
            assert isinstance(model.backbone.posencoder, SinusoidalPositionalEncoding)
        elif posenc == PositionalEncodingArchitecture.LEARNED:
            assert isinstance(model.backbone.posencoder, LearnedPositionalEncoding)
            if parch in (PatcherArchitecture.CNV, PatcherArchitecture.HCV):
                assert max_length is not None
                assert model.backbone.posencoder.max_len == math.ceil(max_length / 4096) + 1
            else:
                assert model.backbone.posencoder.max_len == 256 + 1
        else:
            raise NotImplementedError()

        if patchposenc == PatchPositionalEncodingArchitecture.NONE:
            assert all(isinstance(patchposencoder, Identity) for patchposencoder in model.patchposencoders)
        elif patchposenc == PatchPositionalEncodingArchitecture.REL:
            assert all(isinstance(patchposencoder, PatchPositionalityEncoder) for patchposencoder in model.patchposencoders)
            assert all(patchposencoder.max_length is None for patchposencoder in model.patchposencoders)
        elif patchposenc == PatchPositionalEncodingArchitecture.BTH:
            assert all(isinstance(patchposencoder, PatchPositionalityEncoder) for patchposencoder in model.patchposencoders)
            assert all(patchposencoder.max_length is not None for patchposencoder in model.patchposencoders)
        elif patchposenc == PatchPositionalEncodingArchitecture.ABS:
            assert all(isinstance(patchposencoder, PatchPositionalityEncoder) for patchposencoder in model.patchposencoders)
            assert all(patchposencoder.max_length is not None for patchposencoder in model.patchposencoders)
        else:
            raise NotImplementedError()

        if num_guides == 0:
            assert all(isinstance(filmer, FiLMNoP) for filmer in model.filmers)
        else:
            assert all(isinstance(filmer, FiLM) for filmer in model.filmers)
            assert all(filmer.guide_dim == num_guides for filmer in model.filmers)

        ids_embd = set(id(module) for module in model.embeddings)
        ids_film = set(id(module) for module in model.filmers)
        if share_embeddings:
            assert len(ids_embd) == 1
            assert len(ids_film) == 1
        else:
            assert len(ids_embd) == len(structures)
            assert len(ids_film) == len(structures)

        ids_ptch = set(id(module) for module in model.patchers)
        ids_norm = set(id(module) for module in model.norms)
        ids_ppos = set(id(module) for module in model.patchposencoders)
        if share_patchers:
            assert len(ids_ptch) == 1
            assert len(ids_norm) == 1
            assert len(ids_ppos) == 1
        else:
            assert len(ids_ptch) == len(structures)
            assert len(ids_norm) == len(structures)
            assert len(ids_ppos) == len(structures)
