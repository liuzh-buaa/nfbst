class BestConfig(object):
    def __init__(self):
        self.config = {
            "simulation_v2": ConfigSimulationV2,
            "simulation_v3": ConfigSimulationV3,
            "simulation_v4": ConfigSimulationV4,
            "simulation_v12": ConfigSimulationV12,
            "mnist": ConfigMNIST,
            "cifar10": ConfigCifar10,
            "boston_26": ConfigBoston26,
            "boston_13": ConfigBoston13,
            "concrete_16": ConfigConcrete16,
            "kin8nm_16": ConfigKin8nm16,
            "yacht_12": ConfigYacht12,
            "wine_22": ConfigWine22,
            "power_8": ConfigPower8,
            "naval_y1_32": ConfigNavalY132,
            "naval_y2_32": ConfigNavalY232,
            "protein_18": ConfigProtein18,
            "energy_16": ConfigEnergy16,
            "efficient_16": ConfigEfficient16,
            "energy_16_2": ConfigEnergy162,
            "energy_16_3": ConfigEnergy163,
            "energy_16_4": ConfigEnergy164,
            'energy_8': ConfigEnergy8,
            'energy_9': ConfigEnergy9,
            'energy_8_2': ConfigEnergy82,
            'energy_8_3': ConfigEnergy83
        }

    def get(self, data, model_type):
        if data in self.config.keys():
            return self.config.get(data)().get_with_specific_data(model_type)
        else:
            return None

    def get_with_specific_data(self, model_type):
        return self.config.get(model_type, None)


class ConfigSimulationV2(BestConfig):
    def __init__(self):
        super().__init__()
        self.config = {
            "gaussian": {
                "models_struct": [20, 20, 20],
                "lr": 0.01,
                "lr_steps": 100,
                "lr_gamma": 0.5,
                "beta": 0.01,
                "sigma_pi": 0.1,
                "sigma_start": 0.1
            },
            "nn": {
                "models_struct": [20, 20, 20],
                "lr": 0.01,
                "lr_steps": 30,
                "lr_gamma": 0.5,
                "beta": 0.001
            },
            "xgb": {
                "eta": 0.05,
                "max_depth": 9
            },
            "cat": {
                "learning_rate": None,
                "max_depth": 8
            },
            "lgb": {
                "learning_rate": 0.15,
                "max_depth": 8,
                "num_leaves": 127
            }
        }


class ConfigSimulationV3(BestConfig):
    def __init__(self):
        super().__init__()
        self.config = {
            "gaussian": {
                "models_struct": [20, 20, 20],
                "lr": 0.01,
                "lr_steps": 50,
                "lr_gamma": 0.5,
                "beta": 0.1,
                "sigma_pi": 0.1,
                "sigma_start": 0.1
            },
            "nn": {
                "models_struct": [20, 20, 20],
                "lr": 0.01,
                "lr_steps": 30,
                "lr_gamma": 0.5,
                "beta": 0.01
            }
        }


class ConfigSimulationV4(BestConfig):
    def __init__(self):
        super().__init__()
        self.config = {
            "gaussian": {
                "models_struct": [20, 20, 20],
                "lr": 0.01,
                "lr_steps": 100,
                "lr_gamma": 0.5,
                "beta": 0.01,
                "sigma_pi": 0.1,
                "sigma_start": 0.1
            },
            "nn": {
                "models_struct": [20, 20, 20],
                "lr": 0.01,
                "lr_steps": 100,
                "lr_gamma": 0.5
            },
            "xgb": {
                "eta": None,
                "max_depth": None
            },
            "cat": {
                "learning_rate": None,
                "max_depth": 7
            },
            "lgb": {
                "learning_rate": 0.15,
                "max_depth": -1,
                "num_leaves": 63
            }
        }


class ConfigSimulationV12(BestConfig):
    def __init__(self):
        super().__init__()
        self.config = {
            "gaussian": {
                "models_struct": [20, 20, 20],
                "lr": 0.01,
                "lr_steps": 100,
                "lr_gamma": 0.5,
                "beta": 0.01,
                "sigma_pi": 0.1,
                "sigma_start": 0.1
            },
            "nn": {
                "models_struct": [20, 20, 20],
                "lr": 0.01,
                "lr_steps": 100,
                "lr_gamma": 0.5
            },
            "xgb": {
                "eta": None,
                "max_depth": 6
            },
            "cat": {
                "learning_rate": 0.1,
                "max_depth": None
            },
            "lgb": {
                "learning_rate": 0.15,
                "max_depth": -1,
                "num_leaves": 63
            }
        }


class ConfigMNIST(BestConfig):
    def __init__(self):
        super().__init__()
        self.config = {
            "gaussian": {
                "models_struct": "MnistBNN",
                "lr": 0.001,
                "lr_steps": 10000,
                "beta": 0.001,
                "sigma_pi": 0.01,
                "sigma_start": 0.01
            },
            "nn": {
                "models_struct": "MnistCNN",
                "lr": 0.001,
                "lr_steps": 10000,
            }
        }


class ConfigCifar10(BestConfig):
    def __init__(self):
        super(ConfigCifar10, self).__init__()
        self.config = {
            "gaussian": {
                "models_struct": "vgg16bnn_bn",
                "lr": 0.001,
                "lr_steps": 100,
                "lr_gamma": 0.1,
                "beta": 0.001,
                "sigma_pi": 0.01,
                "sigma_start": 0.01
            },
            "nn": {
                "models_struct": "vgg16_bn",
                "lr": 0.001,
                "lr_steps": 100,
                "lr_gamma": 0.1
            }
        }


class ConfigBoston26(BestConfig):
    def __init__(self):
        super().__init__()
        self.config = {
            "gaussian": {
                "models_struct": [10, 10, 10],
                "lr": 0.1,
                "lr_steps": 10000,
                "lr_gamma": 0.5,
                "beta": 0.1,
                "sigma_pi": 0.01,
                "sigma_start": 0.01
            },
            "nn": {
                "models_struct": [10, 10, 10],
                "lr": 0.05,
                "lr_steps": 10000,
                "lr_gamma": 0.5,
                "beta": 0
            },
            "xgb": {
                "eta": 0.05,
                "max_depth": 5
            },
            "cat": {
                "learning_rate": 0.01,
                "max_depth": 8
            },
            "lgb": {
                "learning_rate": 0.1,
                "num_leaves": 31,
                "max_depth": 6,
            }
        }


class ConfigBoston13(BestConfig):
    def __init__(self):
        super().__init__()
        self.config = {
            "nn": {
                "models_struct": [10, 10, 10],
                "lr": 0.05,
                "lr_steps": 10000,
                "lr_gamma": 0.5,
                "beta": 0
            },
            "gaussian": {
                "models_struct": [10, 10, 10],
                "lr": 0.05,
                "lr_steps": 10000,
                "lr_gamma": 0.5,
                "beta": 0,
                "sigma_pi": 0.1,
                "sigma_start": 0.1
            }
        }


class ConfigConcrete16(BestConfig):
    def __init__(self):
        super().__init__()
        self.config = {
            "gaussian": {
                "models_struct": [20, 20],
                "lr": 0.1,
                "lr_steps": 100,
                "lr_gamma": 0.5,
                "beta": 0.01,
                "sigma_pi": 0.01,
                "sigma_start": 0.01
            },
            "nn": {
                "models_struct": [20, 20],
                "lr": 0.1,
                "lr_steps": 100,
                "lr_gamma": 0.5,
                "beta": 0.0005
            },
            "xgb": {
                "eta": 0.1,
                "max_depth": 4
            },
            "cat": {
                "learning_rate": 0.015,
                "max_depth": None
            },
            "lgb": {
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": 8,
            }
        }


class ConfigKin8nm16(BestConfig):
    def __init__(self):
        super().__init__()
        self.config = {
            "gaussian": {
                "models_struct": [20, 20],
                "lr": 0.001,
                "lr_steps": 10000,
                "lr_gamma": 0.1,
                "beta": 0,
                "sigma_pi": 0.01,
                "sigma_start": 0.01
            },
            "nn": {
                "models_struct": [20, 20],
                "lr": 0.001,
                "lr_steps": 10000,
                "lr_gamma": 0.1,
                "beta": 0
            },
            "xgb": {
                "eta": None,
                "max_depth": None
            },
            "cat": {
                "learning_rate": None,
                "max_depth": None
            },
            "lgb": {
                "learning_rate": 0.1,
                "num_leaves": 127,
                "max_depth": 8,
            }
        }


class ConfigYacht12(BestConfig):
    def __init__(self):
        super().__init__()
        self.config = {
            "gaussian": {
                "models_struct": [20, 20, 20],
                "lr": 0.001,
                "lr_steps": 10000,
                "lr_gamma": 0.1,
                "beta": 0.01,
                "sigma_pi": 0.01,
                "sigma_start": 0.01
            }
        }


class ConfigWine22(BestConfig):
    def __init__(self):
        super().__init__()
        self.config = {
            "gaussian": {
                "models_struct": [20, 20],
                "lr": 0.01,
                "lr_steps": 10000,
                "lr_gamma": 0.1,
                "beta": 0.01,
                "sigma_pi": 0.1,
                "sigma_start": 0.1
            },
            "nn": {
                "models_struct": [20, 20],
                "lr": 0.01,
                "lr_steps": 100,
                "lr_gamma": 0.5,
                "beta": 0,
            },
            "xgb": {
                "eta": 0.05,
                "max_depth": None
            },
            "cat": {
                "learning_rate": None,
                "max_depth": 4
            },
            "lgb": {
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": -1,
            }
        }


class ConfigPower8(BestConfig):
    def __init__(self):
        super().__init__()
        self.config = {
            "gaussian": {
                "models_struct": [20, 20, 20],
                "lr": 0.1,
                "lr_steps": 30,
                "lr_gamma": 0.5,
                "beta": 0.01,
                "sigma_pi": 0.01,
                "sigma_start": 0.01
            },
            "nn": {
                "models_struct": [20, 20, 20],
                "lr": 0.05,
                "lr_steps": 30,
                "lr_gamma": 0.5,
                "beta": 0.01,
            },
            "xgb": {
                "eta": None,
                "max_depth": None
            },
            "cat": {
                "learning_rate": None,
                "max_depth": None,
            },
            "lgb": {
                "learning_rate": 0.1,
                "max_depth": -1,
                "num_leaves": 31
            }
        }


class ConfigNavalY132(BestConfig):
    def __init__(self):
        super().__init__()
        self.config = {
            "gaussian": {
                "models_struct": [20, 20, 20],
                "lr": 0.001,
                "lr_steps": 10000,
                "lr_gamma": 0.1,
                "beta": 0,
                "sigma_pi": 0.05,
                "sigma_start": 0.05
            },
            "nn": {
                "models_struct": [20, 20, 20],
                "lr": 0.01,
                "lr_steps": 30,
                "lr_gamma": 0.5,
                "beta": 0,
            }
        }


class ConfigNavalY232(BestConfig):
    def __init__(self):
        super().__init__()
        self.config = {
            "gaussian": {
                "models_struct": [20, 20, 20],
                "lr": 0.01,
                "lr_steps": 100,
                "lr_gamma": 0.5,
                "beta": 0,
                "sigma_pi": 0.01,
                "sigma_start": 0.01
            },
            "nn": {
                "models_struct": [20, 20, 20],
                "lr": 0.01,
                "lr_steps": 50,
                "lr_gamma": 0.5,
                "beta": 0,
            },
            "xgb": {
                "eta": 0.23,
                "max_depth": 6
            },
            "cat": {
                "learning_rate": 0.07,
                "max_depth": None
            },
            "lgb": {
                "num_leaves": 63,
                "max_depth": 8,
                "learning_rate": 0.15
            }
        }


class ConfigProtein18(BestConfig):
    def __init__(self):
        super().__init__()
        self.config = {
            "gaussian": {
                "models_struct": [20, 20, 20],
                "lr": 0.01,
                "lr_steps": 100,
                "lr_gamma": 0.5,
                "beta": 0.01,
                "sigma_pi": 0.05,
                "sigma_start": 0.05
            }
        }


class ConfigEnergy16(BestConfig):
    def __init__(self):
        super().__init__()
        self.config = {
            "gaussian": {
                "models_struct": [20, 20, 20],
                "lr": 0.05,
                "lr_steps": 10000,
                "lr_gamma": 0.1,
                "beta": 0.1,
                "sigma_pi": 0.1,
                "sigma_start": 0.1
            },
            "nn": {
                "models_struct": [20, 20, 20],
                "lr": 0.05,
                "lr_steps": 100,
                "lr_gamma": 0.1,
                "beta": 0
            },
            "xgb": {
                "eta": 0.038,
                "max_depth": 9
            },
            "cat": {
                "learning_rate": 0.005,
                "max_depth": 5
            },
            "lgb": {
                "learning_rate": 0.042,
                "num_leaves": 31,
                "max_depth": 4,
            }
        }


class ConfigEfficient16(BestConfig):
    def __init__(self):
        super().__init__()
        self.config = {
            "gaussian": {
                "models_struct": [20, 20, 20],
                "lr": 0.1,
                "lr_steps": 100,
                "lr_gamma": 0.5,
                "beta": 0.01,
                "sigma_pi": 0.01,
                "sigma_start": 0.01
            },
            "nn": {
                "models_struct": [20, 20, 20],
                "lr": 0.1,
                "lr_steps": 100,
                "lr_gamma": 0.5,
                "beta": 0.001
            },
            "xgb": {
                "eta": 0.038,
                "max_depth": 9
            },
            "cat": {
                "learning_rate": 0.005,
                "max_depth": 5
            },
            "lgb": {
                "learning_rate": 0.042,
                "num_leaves": 31,
                "max_depth": 4,
            }
        }


class ConfigEnergy162(BestConfig):
    def __init__(self):
        super().__init__()
        self.config = {
            "gaussian": {
                "models_struct": [20, 20, 20],
                "lr": 0.05,
                "lr_steps": 10000,
                "lr_gamma": 0.1,
                "beta": 0.1,
                "sigma_pi": 0.1,
                "sigma_start": 0.1
            },
            "nn": {
                "models_struct": [20, 20, 20],
                "lr": 0.1,
                "lr_steps": 100,
                "lr_gamma": 0.1,
                "beta": 0
            },
        }


class ConfigEnergy163(BestConfig):
    def __init__(self):
        super().__init__()
        self.config = {
            "gaussian": {
                "models_struct": [20, 20, 20],
                "lr": 0.05,
                "lr_steps": 10000,
                "lr_gamma": 0.5,
                "beta": 0.01,
                "sigma_pi": 0.1,
                "sigma_start": 0.1
            },
            "nn": {
                "models_struct": [20, 20, 20],
                "lr": 0.1,
                "lr_steps": 100,
                "lr_gamma": 0.5,
                "beta": 0.01
            },
            "xgb": {
                "eta": 0.04,
                "max_depth": 6
            },
            "cat": {
                "learning_rate": 0.005,
                "max_depth": 6
            },
            "lgb": {
                "learning_rate": 0.042,
                "num_leaves": 31,
                "max_depth": 5,
            }
        }


class ConfigEnergy8(BestConfig):
    def __init__(self):
        super().__init__()
        self.config = {
            "gaussian": {
                "models_struct": [20, 20, 20],
                "lr": 0.1,
                "lr_steps": 100,
                "lr_gamma": 0.5,
                "beta": 0.01,
                "sigma_pi": 0.01,
                "sigma_start": 0.01
            },
            "nn": {
                "models_struct": [20, 20, 20],
                "lr": 0.1,
                "lr_steps": 100,
                "lr_gamma": 0.5,
                "beta": 0.01
            },
        }


class ConfigEnergy82(BestConfig):
    def __init__(self):
        super().__init__()
        self.config = {
            "gaussian": {
                "models_struct": [20, 20, 20],
                "lr": 0.05,
                "lr_steps": 100,
                "lr_gamma": 0.5,
                "beta": 0.001,
                "sigma_pi": 0.001,
                "sigma_start": 0.001
            },
            "nn": {
                "models_struct": [20, 20, 20],
                "lr": 0.01,
                "lr_steps": 100,
                "lr_gamma": 0.5,
                "beta": 0.01
            },
        }


class ConfigEnergy83(BestConfig):
    def __init__(self):
        super().__init__()
        self.config = {
            "gaussian": {
                "models_struct": [20, 20, 20],
                "lr": 0.01,
                "lr_steps": 100,
                "lr_gamma": 0.1,
                "beta": 0.0001,
                "sigma_pi": 0.001,
                "sigma_start": 0.001
            },
            "nn": {
                "models_struct": [20, 20, 20],
                "lr": 0.01,
                "lr_steps": 100,
                "lr_gamma": 0.1,
                "beta": 0.001
            },
        }


class ConfigEnergy9(BestConfig):
    def __init__(self):
        super().__init__()
        self.config = {
            "gaussian": {
                "models_struct": [20, 20, 20],
                "lr": 0.1,
                "lr_steps": 100,
                "lr_gamma": 0.5,
                "beta": 0.01,
                "sigma_pi": 0.01,
                "sigma_start": 0.01
            },
            "nn": {
                "models_struct": [20, 20, 20],
                "lr": 0.1,
                "lr_steps": 100,
                "lr_gamma": 0.5,
                "beta": 0.01
            },
        }


class ConfigEnergy164(BestConfig):
    def __init__(self):
        super().__init__()
        self.config = {
            "gaussian": {
                "models_struct": [20, 20, 20],
                "lr": 0.05,
                "lr_steps": 10000,
                "lr_gamma": 0.1,
                "beta": 0.1,
                "sigma_pi": 0.1,
                "sigma_start": 0.1
            },
        }
