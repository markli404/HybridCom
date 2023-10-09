import torch


class config:
    # frequently used
    DRIFT_TYPE = 'class_swap'    # class_swap guass_noise
    DRITF_PERCENTAGE = None
    CLASS_SWAP = True
    RUN_TYPE = "fedavg" #"case_study"
    CALIBRATION_TYPE = 'pairwise_vaccine'
    # layerwise_vaccine
    # pairwise_vaccine
    # pairwise_vertical_conflicts_avg_cossim1
    # vertical_conflicts_avg_cossim1_iteration

    # pairwise_vertical_conflicts_avg
    # vertical_conflicts_avg_cossim1_iteration_eps3_layerwise
    # vertical_conflicts_avg_cossim1_iteration_dynamic_projection
    RUN_NAME = None
    BASE = None
    SAVE_MODEL = False
    # pairwise_vertical_cossim1
    # iid_fedaverage

    # 'class_intro_our_c_2_0.05_{}'
    RUN_NAME_ALL = 'tsf_20_clients_{}_{}' # 'class_intro_fed_avg_freq_{}_{}' #_coeff0.4_with_reduced_freq=0.15'
    GRADIENT = False
    NUM_CLIENTS = 20
    NUM_ROUNDS = 300
    NUM_TRAINING_SAMPLES = 256                    # number of samples added to local training set
    NUM_TEST_SAMPLES = 128                          # number of samples in the test set
    DRIFT = [600]                                     # when drift happens
    PERCENTAGE = 1
    FRACTION = None                                 # percentage of clients selected each round
    MODEL_COEFF = 1
    NUM_SELECTED_CLASS = 2

    # main
    run_time = 1
    fractions = [1] # [2, 1.7, 1.1]  # 其他的 [0.2, 0.4, 0.6, 0.8, 1]               # fast的频率控制[1.1, 0.8, 0.2, -0.5, -1]
    CACHE = False
    FRESHNESS = False

    # fractions = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
    # fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    SAVE = True

    C_1 = 30
    C_2 = 0.3
    DECAY = 0.15
    # global config
    SEED = 5959
    DEVICE = "cpu"

    # data config
    DATA_PATH = 'data/'
    DATASET_NAME = 'FashionMNIST'
    NUM_CLASS = 10

    # train config
    CRITERION = "torch.nn.CrossEntropyLoss"
    OPTIMIZER = "torch.optim.SGD"
    OPTIMIZER_CONFIG = {
        'lr': 0.01,
        'momentum': 0.9,
    }

    # client config
    LOCAL_EPOCH = 1
    BATCH_SIZE = 256

    # server config
    GLOBAL_TEST_SAMPLES = 2000

    # log config
    LOG_PATH = "./log/"
    LOG_NAME = "FL.log"
    TB_PORT = 5252
    TB_HOST = "0.0.0.0"

    # model config
    # MODEL_NAME = 'TwoNN'
    # MODEL_CONFIG = {
    #     'name': 'TwoNN',
    #     'in_features': 784,
    #     'num_hiddens': 512,
    #     'num_classes': 10,
    # }

    MODEL_NAME = 'CNN'
    MODEL_CONFIG = {
        'name': 'CNN',
        'in_channels': 1,
        'hidden_channels': 32,
        'num_hiddens': 512,
        'num_classes': NUM_CLASS,
    }
    # MODEL_NAME = 'CNN2'
    # MODEL_CONFIG = {
    #     'name': 'CNN2',
    #     'in_channels': 3,
    #     'hidden_channels': 32,
    #     'num_hiddens': 512,
    #     'num_classes': NUM_CLASS,
    # }

    INIT_TYPE = "xavier"
    INIT_GAIN = 1.0
    GPU_IDS = [0]


def load_model():
    return eval(config.MODEL_NAME)(**config.MODEL_CONFIG)