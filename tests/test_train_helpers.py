from gemma_grpo.train import _logit_selection_device, _no_split_module_classes


class _BaseModel:
    _no_split_modules = [{"Gemma4DecoderLayer"}, ("Gemma4TextScaledWordEmbedding",)]


class _PeftWrapper:
    def get_base_model(self):
        return _BaseModel()


def test_no_split_module_classes_flattens_peft_collections():
    assert _no_split_module_classes(_PeftWrapper()) == [
        "Gemma4DecoderLayer",
        "Gemma4TextScaledWordEmbedding",
    ]


class _ParallelBaseModel:
    hf_device_map = {
        "model.language_model.embed_tokens": 0,
        "model.language_model.norm": 3,
        "lm_head": 0,
    }


class _ParallelWrapper:
    def get_base_model(self):
        return _ParallelBaseModel()


def test_logit_selection_device_uses_final_language_norm():
    assert _logit_selection_device(_ParallelWrapper()) == "cuda:3"
