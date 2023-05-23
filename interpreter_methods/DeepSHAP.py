import captum.attr

from interpreter_methods.InterpreterBase import Interpreter


class DeepSHAP(Interpreter):
    def __init__(self, model):
        super().__init__(model)

    def interpret(self, x, target, **kwargs):
        shap = captum.attr.DeepLiftShap(self.model)
        return shap.attribute(x, target=target, **kwargs).data.clone().detach()
