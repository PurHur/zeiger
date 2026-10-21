"""zeiger-lab: the fixed benchmark pipeline for Zeiger architecture and base-model candidates (docs/CONTRACT.md).

Extension points (registries), each a small local diff on an arch/* or base/* branch:
- loss term:          zlab.losses.LOSSES        (@LOSSES.register(name), subclass LossTerm)  -> variant "losses"
- head:               zlab.heads.HEADS          (@HEADS.register(name), nn.Module(width, **cfg)) -> variant "head.kind"
- data transform:     zlab.data.TRANSFORMS      (@TRANSFORMS.register(name), subclass Transform) -> variant "transforms"
- token layout:       zlab.encoding.ENCODINGS   (@ENCODINGS.register(name), fn -> Item)       -> variant "encoding.kind"
- attention per layer: zlab.backbones.ATTENTION (@ATTENTION.register(name), fn(..., ctx))    -> variant "backbone.attention"
- backbone:           zlab.backbones.BACKBONES  (@BACKBONES.register(name), subclass Backbone) -> variant "backbone.kind"
Modules listed in variant "plugins" are imported before the model is built.
"""

__version__ = "0.1.0"
