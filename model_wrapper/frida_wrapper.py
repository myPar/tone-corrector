# Захреначьте импорты сюда и сделайте метод call, как в других обёртках
from FRIDA.model import load_model
from FRIDA.model import generate_resp

class FridaWrapper(object):
    chkp_dir = 'FRIDA/'
    def __init__(self) -> None:
        self.model, self.tokenizer=load_model(FridaWrapper.chkp_dir+"classifier_head.pth")
    
    def __call__(self, text: str) -> str:
        return generate_resp(text,self.model,self.tokenizer)