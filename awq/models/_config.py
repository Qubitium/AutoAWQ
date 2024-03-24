import os
import json
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from transformers.utils.hub import PushToHubMixin, cached_file

_FORMATS = {"gemm", "gemv", "gemmv_fast", "marlin"}


def _normalize_dict(original_cfg: Dict[str: Any]):
    cfg = original_cfg.copy()
    cfg["method"] = cfg.get("method", cfg.get("quant_method", "awq")).lower()
    if cfg["method"] != "awq":
        raise TypeError(f"Unsupported quant method: {cfg.method}")

    cfg["format"] = cfg.get("format", cfg.get("version", "gemm")).lower()
    if cfg["format"] not in _FORMATS:
        raise TypeError(f"Unsupported quant format: {cfg.format}")

    return cfg


@dataclass
class AwqConfig(PushToHubMixin):
    method: str = field(default="awq")
    format: str = field(default="gemm")
    zero_point: bool = field(default=True)
    q_group_size: int = field(default=128)
    w_bit: int = field(default=4)

    config_file_name = "config.json"
    modules_to_not_convert: Optional[List] = None

    @classmethod
    def from_dict(cls, cfg: Dict[str: Any] = None):
        if cfg is None:
            return cls()

        _normalize_dict(cfg)
        return cls(**cfg)

    @classmethod
    def from_pretrained(cls, save_dir: str, **kwargs):
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        commit_hash = kwargs.pop("_commit_hash", None)

        if os.path.isdir(save_dir):  # Local
            resolved_config_file = os.path.join(save_dir, cls.config_file_name)
        else:  # Remote
            resolved_config_file = cached_file(
                save_dir,
                cls.config_file_name,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                use_auth_token=use_auth_token,
                revision=revision,
                local_files_only=local_files_only,
                subfolder=subfolder,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
                _commit_hash=commit_hash,
            )

        quant_config = None
        if os.path.exists(resolved_config_file):
            with open(resolved_config_file, "r", encoding="utf-8") as file:
                loaded_config = json.loads(file.read())

            cfg = loaded_config.get("quantization_config")
            cfg = _normalize_dict(cfg)

            if cfg is not None:
                awq_config = cls.from_transformers_dict(cls, cfg)
                quant_config = cls(**awq_config)

        if quant_config is None:
            quant_config = cls()

        return quant_config

    def to_dict(self):
        return {
            "method": self.method,
            "format": self.format,
            "zero_point": self.zero_point,
            "q_group_size": self.q_group_size,
            "w_bit": self.w_bit,
            "modules_to_not_convert": self.modules_to_not_convert,
        }

    def from_transformers_dict(self, cfg: Dict):
        cfg = _normalize_dict(cfg)

        return {
            "method": cfg.get("method"),
            "format": cfg.get("format"),
            "zero_point": cfg.get("zero_point"),
            "q_group_size": cfg.get("group_size"),
            "w_bit": cfg.get("bits"),
            "modules_to_not_convert": cfg.get("modules_to_not_convert"),
        }
