import captum.attr

from interpreter_methods.InterpreterBase import Interpreter


class DeepLift(Interpreter):
    def __init__(self, model):
        super().__init__(model)

    def interpret(self, x, target, **kwargs):
        dl = captum.attr.DeepLift(self.model)
        return dl.attribute(x, target=target, **kwargs).data.clone().detach()
