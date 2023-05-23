from torch import nn

import interpreter_methods


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.standard = False

    def convert_to_standard_model(self):
        """Note: when you transform the model into a standard model, then remember to recover back if you need."""
        self.standard = True

    def recover_from_standard_model(self):
        """Recover from standard model to double-output model """
        self.standard = False

    def get_interpret(self, x, opt, **kwargs):
        # self.convert_to_standard_model()
        # interpreter = interpreter_methods.interpreter(self, opt.interpret_method)
        interpreter = interpreter_methods.interpreter(self.sample_4_generate_standard_model(), opt.interpret_method)
        x_clone = x.detach().clone().requires_grad_()
        statistic = interpreter.interpret(x_clone, opt.y_index, **kwargs)
        interpreter.release()
        # self.recover_from_standard_model()
        return statistic
