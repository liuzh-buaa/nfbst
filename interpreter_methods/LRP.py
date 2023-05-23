import captum.attr

from interpreter_methods.InterpreterBase import Interpreter


class LRP(Interpreter):
    def __init__(self, model):
        super().__init__(model)

    def interpret(self, x, target):
        lrp = captum.attr.LRP(self.model)
        return lrp.attribute(x, target).data.clone().detach()
