from __future__ import annotations

import os


if os.environ.get("GEMMA_GRPO_VLLM_BNB_COMPAT") == "1":
    from gemma_grpo.vllm_bnb_compat import install_vllm_bnb_compat

    install_vllm_bnb_compat()
