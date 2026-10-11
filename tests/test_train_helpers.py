from gemma_grpo.train import _no_split_module_classes


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
