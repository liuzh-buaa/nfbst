import torch


class Interpreter(object):
    def __init__(self, model):
        self.model = model
        self.handles = []

    def interpret(self, x, target):
        self.model.eval()
        with torch.enable_grad():
            self.model.zero_grad()
            model_output = self.model(x)  # (batch_size, n_target)
            model_output_sum = torch.sum(model_output[:, target])
            model_output_sum.backward()
            return x.grad.data.clone().detach()

    def release(self):
        """
            释放hook和内存，每次计算saliency后都要调用release()
        """
        for handle in self.handles:
            handle.remove()
        for p in self.model.parameters():
            del p.grad
            p.grad = None
        self.handles = []
