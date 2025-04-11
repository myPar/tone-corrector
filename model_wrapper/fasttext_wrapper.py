import fasttext

class FasttextWrapper(object):
    MODEL_PATH: str = "./model_fasttext.bin"

    def __init__(self) -> None:
        self.model = fasttext.load_model(FasttextWrapper.MODEL_PATH)

    def __call__(self, text: str) -> str:
        label, prob=self.model.predict(text)
        return label[0],prob[0]
