import captum.attr

from interpreter_methods.InterpreterBase import Interpreter


class LIME(Interpreter):
    def __init__(self, model):
        super().__init__(model)

    def interpret(self, x, target, **kwargs):
        lime = captum.attr.Lime(self.model)
        return lime.attribute(x, target=target, **kwargs).data.clone().detach()
