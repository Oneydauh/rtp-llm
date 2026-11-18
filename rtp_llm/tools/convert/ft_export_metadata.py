"""Native-independent index writer shared by the offline FT conversion tool."""

import json
import logging
import os
from typing import Mapping, Optional

from safetensors import safe_open


def generate_ft_index(
    output_path: str,
    env_params: Mapping,
    module_metadata: Optional[Mapping] = None,
):
    """Index already exported rank shards without changing their tensor layout.

    The original FT schema and filename remain unchanged. Optional module
    metadata describes offline parameter ownership, not an inference backend.
    """
    files = sorted(
        name for name in os.listdir(output_path) if name.endswith(".safetensors")
    )
    if not files:
        logging.info("No safetensors shards found in %s", output_path)
        return -1
    weight_map = {}
    total_size = 0
    for name in files:
        path = os.path.join(output_path, name)
        total_size += os.path.getsize(path)
        try:
            with safe_open(path, framework="pt") as checkpoint:
                for key in checkpoint.keys():
                    if key in weight_map:
                        logging.warning(
                            "FT tensor %s occurs in %s and %s",
                            key,
                            weight_map[key],
                            name,
                        )
                    weight_map[key] = name
        except Exception as error:
            logging.warning("Cannot read FT shard %s: %s", name, error)
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
        "is_ft_style_weight": True,
        "__env__params__": dict(env_params),
    }
    if module_metadata is not None:
        index["module_export_metadata"] = dict(module_metadata)
    try:
        with open(
            os.path.join(output_path, "model.safetensors.index.json"),
            "w",
            encoding="utf-8",
        ) as output:
            json.dump(index, output, indent=4, ensure_ascii=False)
    except Exception as error:
        logging.info("Could not write FT index: %s", error)
        return -1
