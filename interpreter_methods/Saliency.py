import captum.attr

from interpreter_methods.InterpreterBase import Interpreter


class Saliency(Interpreter):
    def __init__(self, model):
        super().__init__(model)

    def interpret(self, x, target):
        saliency = captum.attr.Saliency(self.model)
        return saliency.attribute(x, target, abs=False).data.clone().detach()
