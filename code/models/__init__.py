"""Model package exposing the manuscript-facing PhyCL-Net aliases."""

from .ams_net_v2 import AMSNetV2

PhyCLNet = AMSNetV2

__all__ = ["AMSNetV2", "PhyCLNet"]
