import configparser
import copy
import logging
import random
import timeit

import numpy as np
import pandas
import torch
from torch.nn import CosineSimilarity
import scipy
from tqdm.auto import tqdm
from collections import OrderedDict
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# custom packages
from .models import *
from .client import Client
from .utils.DriftController import *
from .utils.CommunicationController import *
from .utils.DatasetController import *
from .utils.Printer import *

logger = logging.getLogger(__name__)


class Server(object):
    def __init__(self, writer):
        self._round = 0
        self.clients = None
        self.writer = writer
        self.num_clients = config.NUM_CLIENTS
        self.runtype = config.RUN_TYPE

        self.model = eval(config.MODEL_NAME)(**config.MODEL_CONFIG)

        self.seed = config.SEED
        self.device = config.DEVICE
        self.cos = CosineSimilarity(dim=0, eps=1e-6)

        self.data = None
        self.dataloader = None
        self.round_upload = []
        self.round_accuracy = []

        # drift related
        self.drift_clients = []
        self.source_classes = []
        self.target_classes = []

        # scaffold
        self.c_global = None

        # FGT
        self.sop_cache = None
        self.cos_dict = np.zeros((20, 20))
        self.cos_dict_layerwise = np.zeros((20, 20, 4))
        self.df_sum = pd.DataFrame(data=np.zeros((20, 20)))

    def log(self, message):
        message = f"[Round: {str(self._round).zfill(4)}] " + message
        print(message);
        logging.info(message)
        del message;
        gc.collect()

    def setup(self):
        # initialize weights of the model
        torch.manual_seed(self.seed)
        init_net(self.model)

        self.log(
            f"...successfully initialized model (# parameters: {str(sum(p.numel() for p in self.model.parameters()))})!")

        # initialize DatasetController
        self.DatasetController = DatasetController()
        self.log('...sucessfully initialized dataset controller for [{}]'.format(config.DATASET_NAME))

        # initialize DriftController
        self.DriftController = DriftController(self.DatasetController)
        self.log('...sucessfully initialized drift controller.')

        # create clients
        self.clients = self.create_clients()

        # initialize CommunicationController
        self.CommunicationController = CommunicationController(self.clients)

        # send the model skeleton to all clients
        message = self.CommunicationController.transmit_model(self.model, to_all_clients=True)

        self.log(message)

    def create_clients(self, distribution=None):
        clients = []
        distribution = np.zeros((self.num_clients, config.NUM_CLASS))
        # np.random.seed(config.NUM_SELECTED_CLASS)
        client_idx = np.arange(self.num_clients)
        np.random.shuffle(client_idx)
        for i in client_idx:
            class_pool = np.sum(distribution, axis=0)
            class_pool = np.where(class_pool < int(config.NUM_SELECTED_CLASS * self.num_clients / config.NUM_CLASS))[0]
            try:
                selected_class = np.random.choice(class_pool, config.NUM_SELECTED_CLASS, replace=False)
            except:
                selected_class = np.random.choice(np.arange(config.NUM_CLASS),
                                                  config.NUM_SELECTED_CLASS - len(class_pool), replace=False)
                selected_class = np.append(selected_class, class_pool)
            for j in selected_class:
                distribution[i][j] = 1

        distribution = distribution / config.NUM_SELECTED_CLASS
        for i in range(self.num_clients):
            d = distribution[i]
            client = Client(client_id=i, device=self.device, distribution=distribution[i])
            clients.append(client)

        self.log(f"...successfully created all {str(self.num_clients)} clients!")
        return clients

    def aggregate_models(self, sampled_client_indices, coefficients):
        self.log(f"...with the weights of {str(coefficients)}.")
        new_model = copy.deepcopy(self.model)
        averaged_weights = OrderedDict()

        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].client_current.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]

        self.model.to("cpu")
        for key in self.model.state_dict().keys():
            averaged_weights[key] += self.model.state_dict()[key] * (1 - np.sum(coefficients))
        self.model.to(self.device)

        new_model.load_state_dict(averaged_weights)
        return new_model

    def fedavg_aggregation(self, sampled_client_indices, coeff, eps=0.001):
        # get gradient (delta) of selected clients
        gradients = []
        for i in sampled_client_indices:
            gradient = self.clients[i].get_gradient()
            gradients.append(gradient)
        gradients = np.array(gradients)

        # get sum of gradient
        sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
        new_model = copy.copy(self.model)
        new_model.to('cpu')
        new_weights = new_model.state_dict()
        global_gradient = self.model.unflatten_model(sum_of_gradient)
        for key in new_model.state_dict().keys():
            new_weights[key] = new_weights[key] - 1 * global_gradient[key]

        new_model.load_state_dict(new_weights)

        # client = self.clients[0]
        # client_current = copy.deepcopy(client.client_current)
        # client_current = client_current.state_dict()
        # client_global = copy.deepcopy(client.global_current)
        #
        # averaged_weights = client.get_gradient()
        #
        # new_model = client_global.unflatten_model(averaged_weights)
        # client_global.to('cpu')
        # client_global = client_global.state_dict()
        # for key in client_global.keys():
        #     client_global[key] = client_global[key] - new_model[key]
        #     a = np.subtract(client_global[key], client_current[key])
        #     print(a)

        return new_model

    def aggregate_models_with_cache(self, sampled_client_indices, coeff):
        new_model = copy.deepcopy(self.model)
        averaged_weights = OrderedDict()

        models = []

        for i in range(self.num_clients):
            if i in sampled_client_indices:
                models.append(self.clients[i].client_current)
            else:
                if self.clients[i].client_previous is not None:
                    models.append(self.clients[i].client_previous)

        weight = 1 / len(models)

        for i, model in enumerate(models):
            local_weights = model.state_dict()
            for key in self.model.state_dict().keys():
                if i == 0:
                    averaged_weights[key] = weight * local_weights[key]
                else:
                    averaged_weights[key] += weight * local_weights[key]

        new_model.load_state_dict(averaged_weights)
        return new_model

    def aggregate_models_scaffold(self, sampled_client_indices, coeff):
        total_delta = copy.deepcopy(self.model.state_dict())
        for key in total_delta:
            total_delta[key] = 0.0

        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            c_delta_para = self.clients[idx].c_delta_para
            for key in total_delta:
                total_delta[key] += c_delta_para[key]

        for key in total_delta:
            total_delta[key] = total_delta[key] / len(sampled_client_indices)

        for i in sampled_client_indices:
            client = self.clients[i]
            c_global_para = client.c_global.state_dict()
            for key in c_global_para:
                if c_global_para[key].type() == 'torch.LongTensor':
                    c_global_para[key] += total_delta[key].type(torch.LongTensor)
                elif c_global_para[key].type() == 'torch.cuda.LongTensor':
                    c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
                else:
                    # print(c_global_para[key].type())
                    c_global_para[key] += total_delta[key]

            client.c_global.load_state_dict(c_global_para)
        return self.aggregate_models(sampled_client_indices, coeff)

    # pairwise修正逻辑：对每个客户端 ，检查他和其他客户端的cossim如果<0，就向该客户端做投影，直到所有客户端都校准一次,不进行迭代。
    def aggregate_pairwise_vertical(self, sampled_client_indices, coeff, eps=0.001):
        def gradient_calibration(gradient, project_target):
            projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
            gradient = gradient - projection * project_target
            return gradient

        # get all gredients
        gradients = []
        for i in sampled_client_indices:
            gradient = self.clients[i].get_gradient()
            gradients.append(gradient)
        gradients = np.array(gradients)
        total_conflicts = []
        for i, g in enumerate(gradients):
            # conflicts=[]
            for j, h in enumerate(gradients):
                cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
                if cos_sim < 0:
                    # print('conflict pair %s and %s cossim %s' %(i,j,cos_sim))
                    calibrated_gradient = gradient_calibration(g, h)
                    gradients[i] = calibrated_gradient
                    total_conflicts.append(h)
        print(' %s conficts' % (len(total_conflicts)))
        sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
        new_model = copy.copy(self.model)
        new_model.to('cpu')
        new_weights = new_model.state_dict()
        global_gradient = self.model.unflatten_model(sum_of_gradient)
        for key in new_model.state_dict().keys():
            new_weights[key] = new_weights[key] - 1 * global_gradient[key]

        new_model.load_state_dict(new_weights)
        return new_model

    def aggregate_pairwise_vertical_cossim3(self, sampled_client_indices, coeff, eps=0.001):
        def gradient_calibration(gradient, project_target):
            projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
            gradient = gradient - projection * project_target
            return gradient

        # get all gredients
        gradients = []
        for i in sampled_client_indices:
            gradient = self.clients[i].get_gradient()
            gradients.append(gradient)
        gradients = np.array(gradients)
        total_conflicts = []
        for i, g in enumerate(gradients):
            # conflicts=[]
            for j, h in enumerate(gradients):
                cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
                if cos_sim < -0.3:
                    # print('conflict pair %s and %s cossim %s' %(i,j,cos_sim))
                    calibrated_gradient = gradient_calibration(g, h)
                    gradients[i] = calibrated_gradient
                    total_conflicts.append(h)
        print(' %s conficts' % (len(total_conflicts)))
        sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
        new_model = copy.copy(self.model)
        new_model.to('cpu')
        new_weights = new_model.state_dict()
        global_gradient = self.model.unflatten_model(sum_of_gradient)
        for key in new_model.state_dict().keys():
            new_weights[key] = new_weights[key] - 1 * global_gradient[key]

        new_model.load_state_dict(new_weights)
        return new_model

    def aggregate_pairwise_vertical_cossim1(self, sampled_client_indices, coeff, eps=0.001):
        def gradient_calibration(gradient, project_target):
            projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
            gradient = gradient - projection * project_target
            return gradient

        # get all gredients
        gradients = []
        for i in sampled_client_indices:
            gradient = self.clients[i].get_gradient()
            gradients.append(gradient)
        gradients = np.array(gradients)
        total_conflicts = []
        for i, g in enumerate(gradients):
            # conflicts=[]
            for j, h in enumerate(gradients):
                cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
                if cos_sim < -0.1:
                    # print('conflict pair %s and %s cossim %s' %(i,j,cos_sim))
                    calibrated_gradient = gradient_calibration(g, h)
                    gradients[i] = calibrated_gradient
                    total_conflicts.append(h)
        print(' %s conficts' % (len(total_conflicts)))
        sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
        new_model = copy.copy(self.model)
        new_model.to('cpu')
        new_weights = new_model.state_dict()
        global_gradient = self.model.unflatten_model(sum_of_gradient)
        for key in new_model.state_dict().keys():
            new_weights[key] = new_weights[key] - 1 * global_gradient[key]

        new_model.load_state_dict(new_weights)
        return new_model

    def aggregate_pairwise_vertical_both(self, sampled_client_indices, coeff, eps=0.001):
        def gradient_calibration(gradient, project_target):
            projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
            gradient = gradient - projection * project_target
            return gradient

        # get all gredients
        gradients = []
        for i in sampled_client_indices:
            gradient = self.clients[i].get_gradient()
            gradients.append(gradient)
        gradients = np.array(gradients)
        for i, g in enumerate(gradients):
            for j, h in enumerate(gradients):
                cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
                if cos_sim < 0:
                    print('conflict pair %s and %s cossim %s' % (i, j, cos_sim))
                    calibrated_gradient_i = gradient_calibration(g, h)
                    calibrated_gradient_j = gradient_calibration(h, g)
                    gradients[i] = calibrated_gradient_i
                    gradients[j] = calibrated_gradient_j
        sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
        new_model = copy.copy(self.model)
        new_model.to('cpu')
        new_weights = new_model.state_dict()
        global_gradient = self.model.unflatten_model(sum_of_gradient)
        for key in new_model.state_dict().keys():
            new_weights[key] = new_weights[key] - 1 * global_gradient[key]

        new_model.load_state_dict(new_weights)
        return new_model

    # def aggregate_pairwise_vertical_both(self, sampled_client_indices, coeff, eps=0.001):
    #     def gradient_calibration(gradient, project_target):
    #         projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
    #         gradient = gradient - projection * project_target
    #         return gradient
    #
    #     # get all gredients
    #     gradients = []
    #     for i in sampled_client_indices:
    #         gradient = self.clients[i].get_gradient()
    #         gradients.append(gradient)
    #     gradients = np.array(gradients)
    #     for i, g in enumerate(gradients):
    #         for j, h in enumerate(gradients):
    #             cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
    #             if cos_sim < 0:
    #                 print('conflict pair %s and %s cossim %s' % (i, j, cos_sim))
    #                 calibrated_gradient_i = gradient_calibration(g, h)
    #                 calibrated_gradient_j = gradient_calibration(h, g)
    #                 gradients[i] = calibrated_gradient_i
    #                 gradients[j] = calibrated_gradient_j
    #     sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
    #     new_model = copy.copy(self.model)
    #     new_model.to('cpu')
    #     new_weights = new_model.state_dict()
    #     global_gradient = self.model.unflatten_model(sum_of_gradient)
    #     for key in new_model.state_dict().keys():
    #         new_weights[key] = new_weights[key] - 1 * global_gradient[key]
    #
    #     new_model.load_state_dict(new_weights)
    #     return new_model

    # def aggregate_pairwise_vertical_both_cossim1(self, sampled_client_indices, coeff, eps=0.001):
    #     def gradient_calibration(gradient, project_target):
    #         projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
    #         gradient = gradient - projection * project_target
    #         return gradient
    #
    #     # get all gredients
    #     gradients = []
    #     for i in sampled_client_indices:
    #         gradient = self.clients[i].get_gradient()
    #         gradients.append(gradient)
    #     gradients = np.array(gradients)
    #     for i, g in enumerate(gradients):
    #         for j, h in enumerate(gradients):
    #             cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
    #             if cos_sim < 0:
    #                 print('conflict pair %s and %s cossim %s' % (i, j, cos_sim))
    #                 calibrated_gradient_i = gradient_calibration(g, h)
    #                 calibrated_gradient_j = gradient_calibration(h, g)
    #                 gradients[i] = calibrated_gradient_i
    #                 gradients[j] = calibrated_gradient_j
    #     sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
    #     new_model = copy.copy(self.model)
    #     new_model.to('cpu')
    #     new_weights = new_model.state_dict()
    #     global_gradient = self.model.unflatten_model(sum_of_gradient)
    #     for key in new_model.state_dict().keys():
    #         new_weights[key] = new_weights[key] - 1 * global_gradient[key]
    #
    #     new_model.load_state_dict(new_weights)
    #     return new_model

    def aggregate_pairwise_vertical_both_cossim1(self, sampled_client_indices, coeff, eps=0.001):
        def gradient_calibration(gradient, project_target):
            projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
            gradient = gradient - projection * project_target
            return gradient

        # get all gredients
        gradients = []
        for i in sampled_client_indices:
            gradient = self.clients[i].get_gradient()
            gradients.append(gradient)
        gradients = np.array(gradients)
        for i, g in enumerate(gradients):
            for j, h in enumerate(gradients):
                cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
                if cos_sim < -0.1:
                    # print('conflict pair %s and %s cossim %s' % (i, j, cos_sim))
                    calibrated_gradient_i = gradient_calibration(g, h)
                    calibrated_gradient_j = gradient_calibration(h, g)
                    gradients[i] = calibrated_gradient_i
                    gradients[j] = calibrated_gradient_j
        sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
        new_model = copy.copy(self.model)
        new_model.to('cpu')
        new_weights = new_model.state_dict()
        global_gradient = self.model.unflatten_model(sum_of_gradient)
        for key in new_model.state_dict().keys():
            new_weights[key] = new_weights[key] - 1 * global_gradient[key]

        new_model.load_state_dict(new_weights)
        return new_model

    """The following functions will be rewritten"""

    def get_selected_gradients(self, sampled_client_indices):
        gradients = []
        for i in sampled_client_indices:
            gradient = self.clients[i].get_gradient()
            gradients.append(gradient)
        gradients = np.array(gradients)

        return gradients

    def set_selected_gradients(self, sampled_client_indices, gradients):
        for i, client_id in enumerate(sampled_client_indices):
            self.clients[client_id].set_gradient(gradients[i])

    def gradient_calibration(self, gradient, project_target):
        projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
        gradient = gradient - projection * project_target
        return gradient

    def calibration_pairwise_vertical_conflicts_avg(self, sampled_client_indices, eps=0.001):
        # get all gradients from selected clients
        gradients = self.get_selected_gradients(sampled_client_indices)

        for i, g in enumerate(gradients):
            conflicts = []
            for j, h in enumerate(gradients):
                cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
                if cos_sim < 0:
                    conflicts.append(h)
                    # print('conflict pair %s and %s cossim %s' % (i, j, cos_sim))
                    # calibrated_gradient_i = gradient_calibration(g, h)
                    # calibrated_gradient_j = gradient_calibration(h, g)
                    # gradients[i] = calibrated_gradient_i
                    # gradients[j] = calibrated_gradient_j

            if len(conflicts) > 0:
                print('gradient %s has %s conficts' % (i, len(conflicts)))
                sum_of_conflicts = np.sum(conflicts, axis=0) / len(conflicts)
                calibrated_gradient = self.gradient_calibration(g, sum_of_conflicts)
                gradients[i] = calibrated_gradient

        # restore the client model based on calibration for further aggregation
        self.set_selected_gradients(sampled_client_indices, gradients)

        return gradients

    def calibration_pairwise_vertical_conflicts_avg_cossim1(self, sampled_client_indices, eps=0.001):
        th = 0.95 ** int(self._round / 10)
        print('cossim TH is ', th)

        # get all gradients from selected clients
        gradients = self.get_selected_gradients(sampled_client_indices)

        for i, g in enumerate(gradients):
            conflicts = []
            for j, h in enumerate(gradients):
                cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
                if cos_sim < -0.3 * th:
                    conflicts.append(h)
            if len(conflicts) > 0:
                # print('gradient %s has %s conficts' % (i, len(conflicts)))
                sum_of_conflicts = np.sum(conflicts, axis=0) / len(conflicts)
                calibrated_gradient = self.gradient_calibration(g, sum_of_conflicts)
                gradients[i] = calibrated_gradient

        # restore the client model based on calibration for further aggregation
        self.set_selected_gradients(sampled_client_indices, gradients)

        return gradients

    def calibration_vertical_conflicts_avg_cossim1_iteration(self, sampled_client_indices, eps=0.01):
        th = 0.95 ** int(self._round / 10)
        print('cossim TH is ', th)

        # get all gradients from selected clients
        gradients = self.get_selected_gradients(sampled_client_indices)
        sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)

        cache = None
        if cache is None:
            cache = sum_of_gradient - 2 * eps
        calibration_iteration = 0
        while np.linalg.norm(sum_of_gradient - cache) > eps:
            calibration_iteration += 1
            print("calibration_iteration %s" % calibration_iteration)
            for i, g in enumerate(gradients):
                conflicts = []
                for j, h in enumerate(gradients):
                    cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
                    if cos_sim < -0.3 * th:
                        conflicts.append(h)
                if len(conflicts) > 0:
                    print('gradient %s has %s conficts' % (i, len(conflicts)))
                    sum_of_conflicts = np.sum(conflicts, axis=0) / len(conflicts)
                    calibrated_gradient = self.gradient_calibration(g, sum_of_conflicts)
                    gradients[i] = calibrated_gradient
            cache = sum_of_gradient
            sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
            print("delta of global gradient after calibration is %s" % np.linalg.norm(sum_of_gradient - cache))

        # restore the client model based on calibration for further aggregation
        self.set_selected_gradients(sampled_client_indices, gradients)
        return gradients

    def vertical_conflicts_avg_cossim1_iteration_dynamic_projection(self, sampled_client_indices, coeff, eps=0.03):
        # print('projection weight is', 0.8**int(self._round / 50))
        th = 1.2 ** int(self._round / 50)
        print(' TH is ', th, 'cossim is', -0.3 * th)

        def gradient_calibration(gradient, project_target):
            projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
            gradient = gradient - (1 ** int(self._round / 50)) * projection * project_target
            return gradient

        # get all gredients
        gradients = []
        for i in sampled_client_indices:
            gradient = self.clients[i].get_gradient()
            gradients.append(gradient)
        gradients = np.array(gradients)
        sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
        # sop=sum_of_gradient
        cache = None
        if cache is None:
            cache = sum_of_gradient - 2 * eps
        calibration_iteration = 0
        while np.linalg.norm(sum_of_gradient - cache) > eps:
            calibration_iteration += 1
            # print("calibration_iteration %s" %calibration_iteration)
            for i, g in enumerate(gradients):
                conflicts = []
                for j, h in enumerate(gradients):
                    cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
                    if cos_sim < -0.3 * th:
                        conflicts.append(h)
                if len(conflicts) > 0:
                    # print('gradient %s has %s conficts' % (i, len(conflicts)))
                    sum_of_conflicts = np.sum(conflicts, axis=0) / len(conflicts)
                    calibrated_gradient = gradient_calibration(g, sum_of_conflicts)
                    gradients[i] = calibrated_gradient
            cache = sum_of_gradient
            sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
            # print("delta of global gradient after calibration is %s" %np.linalg.norm(sum_of_gradient - cache))
        new_model = copy.copy(self.model)
        new_model.to('cpu')
        new_weights = new_model.state_dict()
        global_gradient = self.model.unflatten_model(sum_of_gradient)
        for key in new_model.state_dict().keys():
            new_weights[key] = new_weights[key] - 1 * global_gradient[key]

        new_model.load_state_dict(new_weights)
        return new_model

    def calibration_pairwise_vaccine(self, sampled_client_indices, eps=0.03):
        # 用VACCINE方法 判断梯度pair的相似度变化，如果变得差异更大了，就修正。所以是自适应的，没啥超参数了/。
        # 在server端，维护一个客户端cossim的表 cos_dict,缓存历史相似度。对应VACCINE的phiT.
        # 每一次当前cissim和缓存dict cossim比较 如果变得差异更大了（值更小了）就校准，公式按照VACCINE
        # 无论如何都更新cossim。

        # print('projection weight is', 0.8**int(self._round / 50))
        # th = 1.2 ** int(self._round / 50)
        # print(' TH is ', th,'cossim is', -0.3*th)
        # def gradient_calibration(gradient, project_target):
        #     projection = np.dot(gradient, project_target) / np.linalg.norm(project_target)
        #     gradient = gradient - projection * project_target
        #     return gradient
        def gradient_calibration_vaccine(gradient, project_target, i, j, cossim):
            # projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
            projection = (np.linalg.norm(gradient) * self.cos_dict[i, j] * (1 - cossim ** 2) ** 0.5 - cossim * (
                        1 - self.cos_dict[i, j] ** 2) ** 0.5) / np.linalg.norm(project_target) * (
                                     1 - self.cos_dict[i, j] ** 2) ** 0.5
            gradient = gradient + projection * project_target
            return gradient

        # get all gradients from selected clients
        gradients = self.get_selected_gradients(sampled_client_indices)
        # sum_of_gradient=np.sum(gradients, axis=0) / len(gradients)
        # sop=sum_of_gradient
        # cache = None
        # if cache is None:
        #    cache = sum_of_gradient - 2 * eps
        # calibration_iteration=0
        # while np.linalg.norm(sum_of_gradient - cache) > eps:
        #     calibration_iteration+=1
        # print("calibration_iteration %s" %calibration_iteration)
        # df = pd.DataFrame(data=self.cos_dict[0:, 0:])
        for i, g in enumerate(gradients):
            conflicts = []
            conflict_count = 0
            for j, h in enumerate(gradients):
                cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
                if cos_sim < self.cos_dict[i, j]:
                    # conflicts.append(h)
                    calibrated_gradient = gradient_calibration_vaccine(g, h, i, j, cos_sim)
                    gradients[i] = calibrated_gradient
                    conflict_count += 1
                self.cos_dict[i, j] = 0.3 * cos_sim + 0.7 * self.cos_dict[i, j]
            # if conflict_count >0:
            #     print('conflict_count of client %s is %s,'%(i,conflict_count))
            # if len(conflicts)>0:
            #     # print('gradient %s has %s conficts' % (i, len(conflicts)))
            #     #sum_of_conflicts = np.sum(conflicts, axis=0) / len(conflicts)
            #     calibrated_gradient = gradient_calibration(g, sum_of_conflicts,)
            # gradients[i]=calibrated_gradient
        # cache = sum_of_gradient

        # restore the client model based on calibration for further aggregation
        self.set_selected_gradients(sampled_client_indices, gradients)

    def cossim_analysis(self, sampled_client_indices, coeff, eps=0.03):
        # 用VACCINE方法 判断梯度pair的相似度变化，如果变得差异更大了，就修正。所以是自适应的，没啥超参数了/。
        # 在server端，维护一个s客户端cossim的表 cos_dict,缓存历史相似度。对应VACCINE的phiT.
        # 每一次当前cissim和缓存dict cossim比较 如果变得差异更大了（值更小了）就校准，公式按照VACCINE
        # 无论如何都更新cossim。

        # print('projection weight is', 0.8**int(self._round / 50))
        # th = 1.2 ** int(self._round / 50)
        # print(' TH is ', th,'cossim is', -0.3*th)
        # def gradient_calibration(gradient, project_target):
        #     projection = np.dot(gradient, project_target) / np.linalg.norm(project_target)
        #     gradient = gradient - projection * project_target
        #     return gradient
        def gradient_calibration_vaccine(gradient, project_target, i, j, cossim):
            # projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
            projection = (np.linalg.norm(gradient) * self.cos_dict[i, j] * (1 - cossim ** 2) ** 0.5 - cossim * (
                    1 - self.cos_dict[i, j] ** 2) ** 0.5) / np.linalg.norm(project_target) * (
                                 1 - self.cos_dict[i, j] ** 2) ** 0.5
            gradient = gradient + projection * project_target
            return gradient

        # get all gredients
        gradients = []
        for i in sampled_client_indices:
            gradient = self.clients[i].get_gradient()
            gradients.append(gradient)
        gradients = np.array(gradients)
        for i, g in enumerate(gradients):
            conflicts = []
            conflict_count = 0
            for j, h in enumerate(gradients):
                cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
                # if cos_sim < self.cos_dict[i, j]:
                #     # conflicts.append(h)
                #     calibrated_gradient = gradient_calibration_vaccine(g, h, i, j, cos_sim)
                #     gradients[i] = calibrated_gradient
                #     conflict_count += 1
                self.cos_dict[i, j] = cos_sim
            # if conflict_count >0:
            #     print('conflict_count of client %s is %s,'%(i,conflict_count))
            # if len(conflicts)>0:
            #     # print('gradient %s has %s conficts' % (i, len(conflicts)))
            #     #sum_of_conflicts = np.sum(conflicts, axis=0) / len(conflicts)
            #     calibrated_gradient = gradient_calibration(g, sum_of_conflicts,)
            # gradients[i]=calibrated_gradient
        # cache = sum_of_gradient
        sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
        # print("delta of global gradient after calibration is %s" %np.linalg.norm(sum_of_gradient - cache))
        new_model = copy.copy(self.model)
        new_model.to('cpu')
        new_weights = new_model.state_dict()
        global_gradient = self.model.unflatten_model(sum_of_gradient)
        for key in new_model.state_dict().keys():
            new_weights[key] = new_weights[key] - 1 * global_gradient[key]
        new_model.load_state_dict(new_weights)
        if self._round % 30 == 0:
            # print(self.cos_dict)
            df = pd.DataFrame(data=self.cos_dict[0:, 0:])
            self.df_sum = self.df_sum + df
            # print('average heatmap is', self.df_sum)
            # print(df)
            hm = sns.heatmap(df, vmin=-1, vmax=1)
            plt.savefig('./heatmap/pic-class{}-step{}.png'.format(config.NUM_SELECTED_CLASS, self._round))
            plt.show()
            cossim_snapshot = df.mean()
            cossim_snapshot_avg = np.mean(cossim_snapshot)
            # print('cossim_snapshot is' ,cossim_snapshot)
            print('cossim_snapshot mean is', cossim_snapshot_avg)
            # columns=
            # print(df)
        if self._round == config.NUM_ROUNDS:
            # print(self._round )
            # print('average heatmap is',self.df_sum/(config.NUM_ROUNDS/1))
            hm = sns.heatmap(self.df_sum / (config.NUM_ROUNDS / 20), vmin=-1, vmax=1)
            plt.savefig('./heatmap/pic-class{}-final.png'.format(config.NUM_SELECTED_CLASS))
            plt.show()

        return new_model

    def calibration_layerwise_vaccine(self, sampled_client_indices, eps=0.03):
        def get_cossim(h, g):
            cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
            return cos_sim

        def layerwise_calibration(gradient, project_target, i, j):
            gradient = self.model.unflatten_model(gradient)
            project_target = self.model.unflatten_model(project_target)
            for k, key in enumerate(gradient.keys()):
                # print(gradient[key])
                # g_layer = gradient[key].numpy().flatten()
                # sog_layer = project_target[key].numpy().flatten()
                g_layer = gradient[key].flatten()
                sog_layer = project_target[key].flatten()
                # print('flatten',g_layer)
                shape = gradient[key].shape
                cossim = get_cossim(g_layer, sog_layer)
                if cossim < self.cos_dict_layerwise[i, j, k]:
                    projection = (np.linalg.norm(g_layer) * self.cos_dict_layerwise[i, j, k] * (
                                1 - cossim ** 2) ** 0.5 - cossim * (
                                              1 - self.cos_dict_layerwise[i, j, k] ** 2) ** 0.5) / np.linalg.norm(
                        sog_layer) * (1 - self.cos_dict_layerwise[i, j, k] ** 2) ** 0.5
                    g_layer = g_layer + projection * sog_layer
                self.cos_dict_layerwise[i, j, k] = 0.5 * cossim + 0.5 * self.cos_dict_layerwise[i, j, k]
                g_layer = torch.tensor(g_layer.reshape(shape))
                gradient[key] = g_layer
            return self.model.flatten_model(gradient)

        # get all gradients from selected clients
        gradients = self.get_selected_gradients(sampled_client_indices)

        for i, g in enumerate(gradients):
            for j, h in enumerate(gradients):
                calibrated_gradient = layerwise_calibration(g, h, i, j)
                gradients[i] = calibrated_gradient

        # restore the client model based on calibration for further aggregation
        self.set_selected_gradients(sampled_client_indices, gradients)
        return gradients

    # 冲突检测和修正都是layerwise,即每层单独找到冲突的客户端，并投影到平均。
    def vertical_conflicts_avg_cossim1_iteration_eps3_layerwise(self, sampled_client_indices, coeff, eps=0.03):
        # print('cossim TH is ', 0.95 ** int(self._round/10))
        # th=0.95 ** int(self._round/10)
        def gradient_calibration(gradient, project_target):
            projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
            gradient = gradient - projection * project_target
            return gradient

        def get_cossim(h, g):
            cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
            return cos_sim

        def layerwise_calibration(gradient, sum_of_gradient):
            gradient = self.model.unflatten_model(gradient)
            sum_of_gradient = self.model.unflatten_model(sum_of_gradient)
            for key in gradient.keys():
                g_layer = gradient[key].numpy().flatten()
                sog_layer = sum_of_gradient[key].numpy().flatten()
                shape = gradient[key].shape
                cos_sim = get_cossim(g_layer, sog_layer)
                # print(key, cos_sim)
                if cos_sim < 0:
                    # print("cossim of layer %s is %s" %(key,cos_sim))
                    projection = np.dot(g_layer, sog_layer) / np.linalg.norm(sog_layer) ** 2
                    g_layer = g_layer - projection * sog_layer
                g_layer = torch.tensor(g_layer.reshape(shape))
                gradient[key] = g_layer
            return self.model.flatten_model(gradient)

        # def layerwise_detection(j,gradient, h,target):
        #     gradient = self.model.unflatten_model(gradient)
        #     target = self.model.unflatten_model(target)
        #     for key in gradient.keys():
        #         g_layer = gradient[key].numpy().flatten()
        #         sog_layer = target[key].numpy().flatten()
        #         shape = gradient[key].shape
        #         cos_sim = get_cossim(g_layer, sog_layer)
        #         # print(key, cos_sim)
        #         if cos_sim < -0.1:
        #             conflicts.append(h)
        #             projection = np.dot(g_layer, sog_layer) / np.linalg.norm(sog_layer) ** 2
        #             g_layer = g_layer -  projection * sog_layer
        #         g_layer = torch.tensor(g_layer.reshape(shape))
        #         gradient[key] = g_layer
        #     return self.model.flatten_model(gradient)

        # get all gredients
        gradients = []
        for i in sampled_client_indices:
            gradient = self.clients[i].get_gradient()
            gradients.append(gradient)
        gradients = np.array(gradients)
        sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
        # sop=sum_of_gradient
        cache = None
        if cache is None:
            cache = sum_of_gradient - 2 * eps
        calibration_iteration = 0
        while np.linalg.norm(sum_of_gradient - cache) > eps:
            calibration_iteration += 1
            # print("calibration_iteration %s" %calibration_iteration)
            for i, g in enumerate(gradients):
                conflicts = []
                for j, h in enumerate(gradients):
                    cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
                    if cos_sim < -0.3 * 0.9 ** self._round:
                        conflicts.append(h)
                if len(conflicts) > 0:
                    # print('gradient %s has %s conficts' % (i, len(conflicts)))
                    sum_of_conflicts = np.sum(conflicts, axis=0) / len(conflicts)
                    calibrated_gradient = layerwise_calibration(g, sum_of_conflicts)
                    gradients[i] = calibrated_gradient
            cache = sum_of_gradient
            sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
            # print("delta of global gradient after calibration is %s" %np.linalg.norm(sum_of_gradient - cache))

        # restore the client model based on calibration for further aggregation
        self.set_selected_gradients(sampled_client_indices, gradients)

        return new_model

    def vertical_conflicts_avg_cossim1_iteration_eps4(self, sampled_client_indices, coeff, eps=0.04):
        def gradient_calibration(gradient, project_target):
            projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
            gradient = gradient - projection * project_target
            return gradient

        # get all gredients
        gradients = []
        for i in sampled_client_indices:
            gradient = self.clients[i].get_gradient()
            gradients.append(gradient)
        gradients = np.array(gradients)
        sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
        # sop=sum_of_gradient
        cache = None
        if cache is None:
            cache = sum_of_gradient - 2 * eps
        calibration_iteration = 0
        while np.linalg.norm(sum_of_gradient - cache) > eps:
            calibration_iteration += 1
            print("calibration_iteration %s" % calibration_iteration)
            for i, g in enumerate(gradients):
                conflicts = []
                for j, h in enumerate(gradients):
                    cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
                    if cos_sim < -0.1:
                        conflicts.append(h)
                if len(conflicts) > 0:
                    print('gradient %s has %s conficts' % (i, len(conflicts)))
                    sum_of_conflicts = np.sum(conflicts, axis=0) / len(conflicts)
                    calibrated_gradient = gradient_calibration(g, sum_of_conflicts)
                    gradients[i] = calibrated_gradient
            cache = sum_of_gradient
            sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
            print("delta of global gradient after calibration is %s" % np.linalg.norm(sum_of_gradient - cache))
        new_model = copy.copy(self.model)
        new_model.to('cpu')
        new_weights = new_model.state_dict()
        global_gradient = self.model.unflatten_model(sum_of_gradient)
        for key in new_model.state_dict().keys():
            new_weights[key] = new_weights[key] - 1 * global_gradient[key]

        new_model.load_state_dict(new_weights)
        return new_model

    def aggregate_pairwise_vertical_cossim3(self, sampled_client_indices, coeff, eps=0.001):
        def gradient_calibration(gradient, project_target):
            projection = np.dot(gradient, project_target) / np.linalg.norm(project_target) ** 2
            gradient = gradient - projection * project_target
            return gradient

        # get all gredients
        gradients = []
        for i in sampled_client_indices:
            gradient = self.clients[i].get_gradient()
            gradients.append(gradient)
        gradients = np.array(gradients)
        total_conflicts = []
        for i, g in enumerate(gradients):
            # conflicts=[]
            for j, h in enumerate(gradients):
                cos_sim = np.dot(h, g) / (np.linalg.norm(h) * np.linalg.norm(g))
                if cos_sim < -0.3:
                    # print('conflict pair %s and %s cossim %s' %(i,j,cos_sim))
                    calibrated_gradient = gradient_calibration(g, h)
                    gradients[i] = calibrated_gradient
                    total_conflicts.append(h)
        print(' %s conficts' % (len(total_conflicts)))
        sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
        new_model = copy.copy(self.model)
        new_model.to('cpu')
        new_weights = new_model.state_dict()
        global_gradient = self.model.unflatten_model(sum_of_gradient)
        for key in new_model.state_dict().keys():
            new_weights[key] = new_weights[key] - 1 * global_gradient[key]

        new_model.load_state_dict(new_weights)
        return new_model

    def aggregate_models_with_pruning(self, sampled_client_indices, coeff, eps=0.001):
        def gradient_calibration(gradient, sum_of_gradient):
            projection = np.dot(gradient, sum_of_gradient) / np.linalg.norm(sum_of_gradient) ** 2
            if projection > 0:
                gradient = gradient + projection * sum_of_gradient
            else:
                gradient = gradient - 2 * projection * sum_of_gradient
            return gradient

        # get gradient (delta) of selected clients
        gradients = []
        for i in sampled_client_indices:
            gradient = self.clients[i].get_gradient()
            gradients.append(gradient)
        gradients = np.array(gradients)

        # get sum of gradient
        sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
        sop = sum_of_gradient
        if self.sop_cache is None:
            self.sop_cache = sum_of_gradient - 2 * 0.001

        while np.linalg.norm(sop - self.sop_cache) > eps:
            for i, g in enumerate(gradients):
                # g_layers = self.model.unflatten_model(g)
                # sop_layers = self.model.unflatten_model(sop)
                #
                # for key in g_layers.keys():
                #     g_temp = g_layers[key].numpy().flatten()
                #     sop_temp = sop_layers[key].numpy().flatten()
                #     cos_sim = np.dot(g_temp, sop_temp) / (np.linalg.norm(g_temp) * np.linalg.norm(sop_temp))
                #     print(i, cos_sim)

                cos_sim = np.dot(sop, g) / (np.linalg.norm(sop) * np.linalg.norm(g))
                if cos_sim < 0.3:
                    print(i, cos_sim)
                    calibrated_gradient = gradient_calibration(g, sop)
                    gradients[i] = calibrated_gradient
                    cos_sim_new = np.dot(sop, calibrated_gradient) / (
                                np.linalg.norm(sop) * np.linalg.norm(calibrated_gradient))
                    print(i, cos_sim_new)

            self.sop_cache = sum_of_gradient
            sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
            sop = sum_of_gradient

        new_model = copy.copy(self.model)
        new_model.to('cpu')
        new_weights = new_model.state_dict()
        global_gradient = self.model.unflatten_model(sum_of_gradient)
        for key in new_model.state_dict().keys():
            new_weights[key] = new_weights[key] - 1 * global_gradient[key]

        new_model.load_state_dict(new_weights)

        # client = self.clients[0]
        # client_current = copy.deepcopy(client.client_current)
        # client_current = client_current.state_dict()
        # client_global = copy.deepcopy(client.global_current)
        #
        # averaged_weights = client.get_gradient()
        #
        # new_model = client_global.unflatten_model(averaged_weights)
        # client_global.to('cpu')
        # client_global = client_global.state_dict()
        # for key in client_global.keys():
        #     client_global[key] = client_global[key] - new_model[key]
        #     a = np.subtract(client_global[key], client_current[key])
        #     print(a)

        return new_model

    def aggregate_models_with_pruning_nagetive_cossim_only(self, sampled_client_indices, coeff, eps=0.001):
        def gradient_calibration(gradient, sum_of_gradient):
            projection = np.dot(gradient, sum_of_gradient) / np.linalg.norm(sum_of_gradient) ** 2
            gradient = gradient - 2 * projection * sum_of_gradient
            return gradient

        # get gradient (delta) of selected clients
        gradients = []
        for i in sampled_client_indices:
            gradient = self.clients[i].get_gradient()
            gradients.append(gradient)
        gradients = np.array(gradients)

        # get sum of gradient
        sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
        sop = sum_of_gradient
        if self.sop_cache is None:
            self.sop_cache = sum_of_gradient - 2 * 0.001

        while np.linalg.norm(sop - self.sop_cache) > eps:
            for i, g in enumerate(gradients):
                # g_layers = self.model.unflatten_model(g)
                # sop_layers = self.model.unflatten_model(sop)
                #
                # for key in g_layers.keys():
                #     g_temp = g_layers[key].numpy().flatten()
                #     sop_temp = sop_layers[key].numpy().flatten()
                #     cos_sim = np.dot(g_temp, sop_temp) / (np.linalg.norm(g_temp) * np.linalg.norm(sop_temp))
                #     print(i, cos_sim)

                cos_sim = np.dot(sop, g) / (np.linalg.norm(sop) * np.linalg.norm(g))
                if cos_sim < 0:
                    print(i, 'cossim before projection', cos_sim)
                    calibrated_gradient = gradient_calibration(g, sop)
                    gradients[i] = calibrated_gradient
                    cos_sim_new = np.dot(sop, calibrated_gradient) / (
                                np.linalg.norm(sop) * np.linalg.norm(calibrated_gradient))
                    print(i, 'cossim after projection', cos_sim_new)

            self.sop_cache = sum_of_gradient
            sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
            sop = sum_of_gradient

        new_model = copy.copy(self.model)
        new_model.to('cpu')
        new_weights = new_model.state_dict()
        global_gradient = self.model.unflatten_model(sum_of_gradient)
        for key in new_model.state_dict().keys():
            new_weights[key] = new_weights[key] - 1 * global_gradient[key]

        new_model.load_state_dict(new_weights)

        # client = self.clients[0]
        # client_current = copy.deepcopy(client.client_current)
        # client_current = client_current.state_dict()
        # client_global = copy.deepcopy(client.global_current)
        #
        # averaged_weights = client.get_gradient()
        #
        # new_model = client_global.unflatten_model(averaged_weights)
        # client_global.to('cpu')
        # client_global = client_global.state_dict()
        # for key in client_global.keys():
        #     client_global[key] = client_global[key] - new_model[key]
        #     a = np.subtract(client_global[key], client_current[key])
        #     print(a)

        return new_model

    def aggregate_models_with_pruning_nagetive_cossim_vertical_cossim1(self, sampled_client_indices, coeff, eps=0.001):
        def gradient_calibration(gradient, sum_of_gradient):
            projection = np.dot(gradient, sum_of_gradient) / np.linalg.norm(sum_of_gradient) ** 2
            gradient = gradient - projection * sum_of_gradient
            return gradient

        # get gradient (delta) of selected clients
        gradients = []
        for i in sampled_client_indices:
            gradient = self.clients[i].get_gradient()
            gradients.append(gradient)
        gradients = np.array(gradients)

        # get sum of gradient
        sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
        sop = sum_of_gradient
        if self.sop_cache is None:
            self.sop_cache = sum_of_gradient - 2 * 0.001

        while np.linalg.norm(sop - self.sop_cache) > eps:
            for i, g in enumerate(gradients):
                cos_sim = np.dot(sop, g) / (np.linalg.norm(sop) * np.linalg.norm(g))
                if cos_sim < -0.1:
                    print(i, 'cossim before projection', cos_sim)
                    calibrated_gradient = gradient_calibration(g, sop)
                    gradients[i] = calibrated_gradient
                    cos_sim_new = np.dot(sop, calibrated_gradient) / (
                                np.linalg.norm(sop) * np.linalg.norm(calibrated_gradient))
                    print(i, 'cossim after projection', cos_sim_new)

            self.sop_cache = sum_of_gradient
            sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
            sop = sum_of_gradient

        new_model = copy.copy(self.model)
        new_model.to('cpu')
        new_weights = new_model.state_dict()
        global_gradient = self.model.unflatten_model(sum_of_gradient)
        for key in new_model.state_dict().keys():
            new_weights[key] = new_weights[key] - 1 * global_gradient[key]

        new_model.load_state_dict(new_weights)
        return new_model

    def aggregate_models_with_pruning_nagetive_cossim_only_vertical(self, sampled_client_indices, coeff, eps=0.001):
        def gradient_calibration(gradient, sum_of_gradient):
            projection = np.dot(gradient, sum_of_gradient) / np.linalg.norm(sum_of_gradient) ** 2
            gradient = gradient - projection * sum_of_gradient
            return gradient

        # get gradient (delta) of selected clients
        gradients = []
        for i in sampled_client_indices:
            gradient = self.clients[i].get_gradient()
            gradients.append(gradient)
        gradients = np.array(gradients)

        # get sum of gradient
        sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
        sop = sum_of_gradient
        if self.sop_cache is None:
            self.sop_cache = sum_of_gradient - 2 * 0.001

        while np.linalg.norm(sop - self.sop_cache) > eps:
            for i, g in enumerate(gradients):
                cos_sim = np.dot(sop, g) / (np.linalg.norm(sop) * np.linalg.norm(g))
                if cos_sim < 0:
                    print(i, 'cossim before projection', cos_sim)
                    calibrated_gradient = gradient_calibration(g, sop)
                    gradients[i] = calibrated_gradient
                    cos_sim_new = np.dot(sop, calibrated_gradient) / (
                                np.linalg.norm(sop) * np.linalg.norm(calibrated_gradient))
                    print(i, 'cossim after projection', cos_sim_new)
            self.sop_cache = sum_of_gradient
            sum_of_gradient = np.sum(gradients, axis=0) / len(gradients)
            sop = sum_of_gradient

        new_model = copy.copy(self.model)
        new_model.to('cpu')
        new_weights = new_model.state_dict()
        global_gradient = self.model.unflatten_model(sum_of_gradient)
        for key in new_model.state_dict().keys():
            new_weights[key] = new_weights[key] - 1 * global_gradient[key]

        new_model.load_state_dict(new_weights)

        # client = self.clients[0]
        # client_current = copy.deepcopy(client.client_current)
        # client_current = client_current.state_dict()
        # client_global = copy.deepcopy(client.global_current)
        #
        # averaged_weights = client.get_gradient()
        #
        # new_model = client_global.unflatten_model(averaged_weights)
        # client_global.to('cpu')
        # client_global = client_global.state_dict()
        # for key in client_global.keys():
        #     client_global[key] = client_global[key] - new_model[key]
        #     a = np.subtract(client_global[key], client_current[key])
        #     print(a)

        return new_model

    def calculate_similarity(self, model_1, model_2):
        # tensor_1 = model_1.flatten_model()
        tensor_1 = model_1
        tensor_2 = model_2.flatten_model()
        assert (tensor_1.shape[0] == tensor_2.shape[0])

        return self.cos(tensor_1, tensor_2).numpy().item()

    def get_perfect_coeff_LS(self, sampled_client_indices):
        D = []
        for i in sampled_client_indices:
            client = self.clients[i]
            D.append(client.distribution)

        penalty = 0.3

        A = np.array(D).T
        I = np.identity(len(sampled_client_indices)) * penalty
        A = np.vstack([A, I])

        b = np.ones(config.NUM_CLASS) / config.NUM_CLASS * len(sampled_client_indices)
        p = np.ones(len(sampled_client_indices)) * penalty
        b = np.hstack([b, p])

        coeff = scipy.optimize.nnls(A, b)[0]
        coeff = coeff / np.sum(coeff) * len(sampled_client_indices) / self.num_clients

        d = np.matmul(np.array(D).T, coeff)
        print(d)
        return coeff

    def get_perfect_coeff_MC(self, sampled_client_indices):
        A = []
        for i in sampled_client_indices:
            client = self.clients[i]
            A.append(client.distribution)

        A = np.array(A).T
        target_coeff = 1 / self.num_clients
        target_distribution = np.ones(config.NUM_CLASS) / config.NUM_CLASS
        best_coeff = None
        best_residual = 100000
        best_distribution = None
        for i in range(1000000):
            coeff = [random.uniform(target_coeff * 0.4, target_coeff * 1.6) for j in range(len(sampled_client_indices))]
            current_distribution = np.matmul(A, coeff)
            current_distribution = current_distribution / np.sum(current_distribution)
            residual = np.linalg.norm(current_distribution - target_distribution)
            if residual < best_residual:
                best_residual = residual
                best_coeff = coeff
                best_distribution = current_distribution
        best_coeff = best_coeff / np.sum(best_coeff) * len(sampled_client_indices) / self.num_clients
        return best_coeff

    def get_blanced_coeff(self, sampled_client_indices):
        num_of_drifts = 0
        for i in sampled_client_indices:
            if self.clients[i].drift:
                num_of_drifts += 1

        coeff = []
        for i in sampled_client_indices:
            if self.clients[i].drift:
                coeff.append(1 / num_of_drifts)
            else:
                coeff.append(1 / (len(sampled_client_indices) - num_of_drifts))

        coeff = coeff / np.sum(coeff) * len(sampled_client_indices) / self.num_clients
        return coeff

    def get_uniformed_coeff(self, sampled_client_indices):
        return np.ones(len(sampled_client_indices)) / config.NUM_CLIENTS

    def calibration_updates(self, sampled_client_indices):
        if config.CALIBRATION_TYPE == 'pairwise_vertical_conflicts_avg':
            self.calibration_pairwise_vertical_conflicts_avg(sampled_client_indices)
        elif config.CALIBRATION_TYPE == 'pairwise_vaccine':
            self.calibration_pairwise_vaccine(sampled_client_indices)
        elif config.CALIBRATION_TYPE == 'layerwise_vaccine':
            self.calibration_layerwise_vaccine(sampled_client_indices)
        else:
            return

    def update_model(self, sampled_client_indices, coeff_method, update_method):
        """Average the updated and transmitted parameters from each selected client."""
        if not sampled_client_indices:
            message = f"None of the clients were selected"
            self.round_upload.append(0)
            self.log(message)
            return

        message = f"Updating {sampled_client_indices} clients...!"
        self.log(message)
        self.round_upload.append(len(sampled_client_indices))

        coeff = coeff_method(sampled_client_indices)
        self.log(message)
        self.model = update_method(sampled_client_indices, coeff)

    def train_without_drift(self, sample_method, coeff_method, update_method, update_type=config.RUN_TYPE):
        # assign new training and test set based on distribution
        self.DriftController.enforce_drift(self.clients, None)
        # train all clients model with local dataset
        message = self.CommunicationController.update_selected_clients(update_type, all_client=True)
        self.log(message)

        message, sampled_client_indices = sample_method()

        self.log(message)
        # evaluate selected clients with local dataset
        # message = self.CommunicationController.evaluate_all_models()
        # self.log(message)

        # update model parameters of the selected clients and update the global model
        self.update_model(sampled_client_indices, coeff_method, update_method)

        # send global model to the selected clients
        message = self.CommunicationController.transmit_model(self.model)
        self.log(message)

    def train_with_drift_after_converge(self, sample_method, coeff_method, update_method, update_type=config.RUN_TYPE,
                                        drift_type=config.DRIFT_TYPE):
        # assign new training and test set based on distribution
        self.DriftController.enforce_drift(self.clients, drift_type)

        # train all clients model with local dataset
        message = self.CommunicationController.update_selected_clients(update_type, all_client=True)
        self.log(message)

        # select clients based on our sample_method
        message, sampled_client_indices = sample_method()

        # update selection related parameters
        for client in self.clients:
            if client.id in sampled_client_indices:
                client.global_previous = client.global_current
                client.client_previous = client.client_current
                client.test_previous = client.test
                client.freshness = 1
            else:
                client.freshness -= 0.4

        self.log(message)

        # update the client local model with calibration if needed
        self.calibration_updates(sampled_client_indices)

        # evaluate selected clients with local dataset
        # message = self.CommunicationController.evaluate_all_models()
        # self.log(message)

        # update model parameters of the selected clients and update the global model
        self.update_model(sampled_client_indices, coeff_method, update_method)

        # send global model to the selected clients
        message = self.CommunicationController.transmit_model(self.model)
        self.log(message)

    def train_fedavg(self):
        """Do federated training."""
        # assign new training and test set based on distribution
        self.enforce_drift()
        self.DatasetController.update_clients_datasets(self.clients, self.drift_clients, self.source_classes,
                                                       self.target_classes, True)

        # train all clients model with local dataset
        message = self.CommunicationController.update_selected_clients(all_client=True)
        self.log(message)

        # select clients based on weights
        if self._round < 15:
            message, sampled_client_indices = self.CommunicationController.sample_all_clients()
        else:
            message, sampled_client_indices = self.CommunicationController.sample_clients_random()
        self.log(message)

        # evaluate selected clients with local dataset
        # message = self.CommunicationController.evaluate_selected_models()
        # self.log(message)

        # update model parameters of the selected clients and update the global model
        self.update_model(sampled_client_indices)

        # send global model to the selected clients
        message = self.CommunicationController.transmit_model(self.model)
        self.log(message)

    def evaluate_global_model(self):
        """Evaluate the global model using the global holdout dataset (self.data)."""
        # calculate the sample distribution of all clients
        global_distribution, global_test_set = self.get_test_dataset()

        message = pretty_list(global_distribution)
        self.log(f"Current test set distribution: [{str(message)}]. ")
        # start evaluation process
        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0
        correct_per_class = np.zeros(config.NUM_CLASS)
        total_per_class = np.zeros(config.NUM_CLASS)
        with torch.no_grad():
            for data, labels in global_test_set.get_dataloader():
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                test_loss += eval(config.CRITERION)()(outputs, labels).item()

                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                labels = labels.cpu().numpy()
                predicted = predicted.cpu().numpy().flatten()
                for i in range(config.NUM_CLASS):
                    c = np.where(labels == i)[0].tolist()
                    if not c:
                        continue
                    total_per_class[i] += len(c)
                    predicted_i = predicted[c]
                    predicted_correct = np.where(predicted_i == i)[0]
                    correct_per_class[i] += len(predicted_correct)

                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

        class_accuracy = []
        for i in range(len(total_per_class)):
            try:
                class_accuracy.append(correct_per_class[i] / total_per_class[i])
            except:
                class_accuracy.append(0)
        class_accuracy = ["%.2f" % i for i in class_accuracy]

        # calculate the metrics
        test_loss = test_loss / len(global_test_set.get_dataloader())
        test_accuracy = correct / len(global_test_set)
        self.round_accuracy.append(test_accuracy)

        # print to tensorboard and log
        self.writer.add_scalar('Loss', test_loss, self._round)
        self.writer.add_scalar('Accuracy', test_accuracy, self._round)

        message = f"Evaluate global model's performance...!\
                \n\t[Server] ...finished evaluation!\
                \n\t=> Loss: {test_loss:.4f}\
                \n\t=> Accuracy: {100. * test_accuracy:.2f}%\
                \n\t=> Class Accuracy: {class_accuracy}\n"
        self.log(message)

    def update_client_distribution(self, distribution, addition=False, everyone=False):
        for client in self.clients:
            if client.drift or everyone:
                if addition:
                    client.distribution += distribution
                else:
                    client.distribution = distribution

                self.log(f"Client {str(client.id)} has a shifted distribution: {str(client.distribution)}")

    def get_test_dataset(self):
        global_distribution = np.zeros(config.NUM_CLASS)
        for client in self.clients:
            global_distribution += client.distribution
        global_distribution = global_distribution / sum(global_distribution)

        global_test_set = None
        for client in self.clients:
            if global_test_set is None:
                global_test_set = copy.deepcopy(client.test)
            else:
                global_test_set + client.test

        return global_distribution, global_test_set

    def save_model(self):
        path = os.path.join('models', self.runtype)
        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(path, self.runtype + '_' + str(self._round) + '.pth')
        torch.save({'model': self.model.state_dict()}, path)

    def class_swap(self, client, class_1=1, class_2=2):
        client.train.class_swap(class_1, class_2)
        client.test.class_swap(class_1, class_2)

    def fit(self):
        """Execute the whole process of the federated learning."""
        for r in range(config.NUM_ROUNDS):
            self._round = r + 1

            # train the model
            """
            train method takes following parameters
                sample_method: method used to select client in the update
                coeff_method: method used to calculate coeff for model aggregation
                    get_uniformed_coeff(sampled_client_indices)
            """
            if False:  # self._round < config.DRIFT[0]:
                self.train_with_drift_after_converge(
                    self.CommunicationController.sample_all_clients,
                    self.get_uniformed_coeff,
                    self.aggregate_models, update_type='fedavg', drift_type='None')
            elif config.RUN_TYPE == 'fedavg':
                self.train_with_drift_after_converge(
                    self.CommunicationController.sample_clients,
                    self.get_uniformed_coeff,
                    self.aggregate_models)
            elif config.RUN_TYPE in ['case_study', 'case_study_test']:
                self.train_with_drift_after_converge(
                    self.CommunicationController.sample_clients_casestudy,
                    self.get_uniformed_coeff,
                    self.aggregate_models)
            elif config.RUN_TYPE == 'fedpns':
                self.train_with_drift_after_converge(
                    self.CommunicationController.sample_clients_fed_pns,
                    self.get_uniformed_coeff,
                    self.aggregate_models)
            elif config.RUN_TYPE == 'fast':
                self.train_with_drift_after_converge(
                    self.CommunicationController.sample_clients_TSF,
                    self.get_perfect_coeff_LS,
                    self.aggregate_models)
            elif config.RUN_TYPE == 'fedprox':
                self.train_with_drift_after_converge(
                    self.CommunicationController.sample_clients,
                    self.get_uniformed_coeff,
                    self.aggregate_models)
            elif config.RUN_TYPE == 'scaffold':
                self.train_with_drift_after_converge(
                    self.CommunicationController.sample_clients,
                    self.get_uniformed_coeff,
                    self.aggregate_models_scaffold)
            elif config.RUN_TYPE == 'new':
                self.train_without_drift(self.CommunicationController.sample_all_clients,
                                         self.get_uniformed_coeff,
                                         self.aggregate_models_with_pruning)
            elif config.RUN_TYPE == 'neg_cossim':
                self.train_without_drift(self.CommunicationController.sample_all_clients,
                                         self.get_uniformed_coeff,
                                         self.aggregate_models_with_pruning_nagetive_cossim_only)
            elif config.RUN_TYPE == 'neg_cossim_vertical':
                self.train_without_drift(self.CommunicationController.sample_all_clients,
                                         self.get_uniformed_coeff,
                                         self.aggregate_models_with_pruning_nagetive_cossim_only_vertical)
            elif config.RUN_TYPE == 'pairwise_vertical':
                self.train_without_drift(self.CommunicationController.sample_all_clients,
                                         self.get_uniformed_coeff,
                                         self.aggregate_pairwise_vertical)
            elif config.RUN_TYPE == 'pairwise_vertical_cossim3':
                self.train_without_drift(self.CommunicationController.sample_all_clients,
                                         self.get_uniformed_coeff,
                                         self.aggregate_pairwise_vertical_cossim3)
            elif config.RUN_TYPE == 'pairwise_vertical_cossim1':
                self.train_without_drift(self.CommunicationController.sample_all_clients,
                                         self.get_uniformed_coeff,
                                         self.aggregate_pairwise_vertical_cossim1)
            elif config.RUN_TYPE == 'pairwise_vertical_both':
                self.train_without_drift(self.CommunicationController.sample_all_clients,
                                         self.get_uniformed_coeff,
                                         self.aggregate_pairwise_vertical_both)
            elif config.RUN_TYPE == 'aggregate_pairwise_vertical_both_cossim1':
                self.train_without_drift(self.CommunicationController.sample_all_clients,
                                         self.get_uniformed_coeff,
                                         self.aggregate_pairwise_vertical_both_cossim1)
            elif config.RUN_TYPE == 'fedaverage':
                self.train_without_drift(self.CommunicationController.sample_all_clients,
                                         self.get_uniformed_coeff,
                                         self.fedavg_aggregation)
            #     每个客户端梯度向冲突梯度的average投影，避免多次投影的覆盖问题。
            elif config.RUN_TYPE == 'pairwise_vertical_conflicts_avg':
                self.train_without_drift(self.CommunicationController.sample_all_clients,
                                         self.get_uniformed_coeff,
                                         self.aggregate_pairwise_vertical_conflicts_avg)

                # aggregate_models_with_pruning_nagetive_cossim_vertical_cossim1
            elif config.RUN_TYPE == 'SOP_vertical_cossim1':
                self.train_without_drift(self.CommunicationController.sample_all_clients,
                                         self.get_uniformed_coeff,
                                         self.aggregate_models_with_pruning_nagetive_cossim_vertical_cossim1)
                # aggregate_pairwise_vertical_conflicts_avg_cossim1
            elif config.RUN_TYPE == 'pairwise_vertical_conflicts_avg_cossim1':
                self.train_without_drift(self.CommunicationController.sample_all_clients,
                                         self.get_uniformed_coeff,
                                         self.aggregate_pairwise_vertical_conflicts_avg_cossim1)
                # vertical_conflicts_avg_cossim1_iteration
            elif config.RUN_TYPE == 'vertical_conflicts_avg_cossim1_iteration':
                self.train_without_drift(self.CommunicationController.sample_all_clients,
                                         self.get_uniformed_coeff,
                                         self.vertical_conflicts_avg_cossim1_iteration)
                # vertical_conflicts_avg_cossim1_iteration_eps3
            elif config.RUN_TYPE == 'vertical_conflicts_avg_cossim1_iteration_eps3':
                self.train_without_drift(self.CommunicationController.sample_all_clients,
                                         self.get_uniformed_coeff,
                                         self.vertical_conflicts_avg_cossim1_iteration_eps3)
            elif config.RUN_TYPE == 'vertical_conflicts_avg_cossim1_iteration_eps4':
                self.train_without_drift(self.CommunicationController.sample_all_clients,
                                         self.get_uniformed_coeff,
                                         self.vertical_conflicts_avg_cossim1_iteration_eps4)
            # vertical_conflicts_avg_cossim1_iteration_eps3_layerwise
            elif config.RUN_TYPE == 'vertical_conflicts_avg_cossim1_iteration_eps3_layerwise':
                self.train_without_drift(self.CommunicationController.sample_all_clients,
                                         self.get_uniformed_coeff,
                                         self.vertical_conflicts_avg_cossim1_iteration_eps3_layerwise)
            # vertical_conflicts_avg_cossim1_iteration_dynamic_projection
            elif config.RUN_TYPE == 'vertical_conflicts_avg_cossim1_iteration_dynamic_projection':
                self.train_without_drift(self.CommunicationController.sample_all_clients,
                                         self.get_uniformed_coeff,
                                         self.vertical_conflicts_avg_cossim1_iteration_dynamic_projection)
            elif config.RUN_TYPE == 'layerwise_vaccine':
                self.train_without_drift(self.CommunicationController.sample_all_clients,
                                         self.get_uniformed_coeff,
                                         self.layerwise_vaccine)
                # pairwise_vaccine
            elif config.RUN_TYPE == 'pairwise_vaccine':
                self.train_without_drift(self.CommunicationController.sample_all_clients,
                                         self.get_uniformed_coeff,
                                         self.pairwise_vaccine)
            elif config.RUN_TYPE == 'cossim_analysis':
                self.train_without_drift(self.CommunicationController.sample_all_clients,
                                         self.get_uniformed_coeff,
                                         self.cossim_analysis)



            else:
                raise Exception("No federal learning method is found.")

            if config.SAVE_MODEL:
                if r % 5 == 0:
                    self.save_model()

            # evaluate the model
            self.evaluate_global_model()

            message = f"Clients have uploaded their model {str(sum(self.round_upload))} times！"
            self.log(message)

            message = f"Overall Accuracy is {str(sum(self.round_accuracy) / len(self.round_accuracy))}!"
            self.log(message)

        self.writer.add_text('accuracy', str(sum(self.round_accuracy) / len(self.round_accuracy)))
        self.writer.add_text('freq', str(sum(self.round_upload)))

        return self.round_accuracy, self.round_upload
