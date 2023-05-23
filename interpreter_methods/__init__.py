def interpreter(model, method='gradient'):
    if method == 'gradient':
        from interpreter_methods.InterpreterBase import Interpreter
        return Interpreter(model)
    elif method == 'Saliency':
        from interpreter_methods.Saliency import Saliency
        return Saliency(model)
    elif method == 'DeepLIFT':
        from interpreter_methods.DeepLIFT import DeepLift
        return DeepLift(model)
    elif method == 'DeepSHAP':
        from interpreter_methods.DeepSHAP import DeepSHAP
        return DeepSHAP(model)
    elif method == 'LRP':
        from interpreter_methods.LRP import LRP
        return LRP(model)
    elif method == 'LIME':
        from interpreter_methods.LIME import LIME
        return LIME(model)
    else:
        raise NotImplementedError(f'No such a interpret method of {method}')
