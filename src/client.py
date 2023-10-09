import gc
import copy
from collections import OrderedDict

# custom packages
from .config import config
from .utils.DatasetController import *

logger = logging.getLogger(__name__)


class Client(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        train: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        client_current: torch.nn instance as a local model.
    """

    def __init__(self, client_id, device, distribution):
        """client training configs"""
        self.batch_size = config.BATCH_SIZE
        self.local_epoch = config.LOCAL_EPOCH
        self.criterion = config.CRITERION
        self.optimizer = config.OPTIMIZER
        self.optim_config = config.OPTIMIZER_CONFIG

        """server side configs"""
        self.id = client_id
        self.device = device
        self.distribution = distribution
        self.drift = False

        # dataset
        self.train = None
        self.test = None
        self.test_previous = None

        # models
        self.time_s = 1
        self.freshness = 1
        self.client_current = None
        self.global_previous = None
        self.client_previous = None
        self.global_current = None

        # For scaffold only
        self.c_local = None
        self.c_global = None
        self.c_delta = None

    def get_gradient(self):
        grad = np.subtract(self.global_current.flatten_model(), self.client_current.flatten_model())
        # return grad / (args.num_sample * args.local_ep * lr / args.local_bs)
        # grad = grad / (len(self.train) * self.local_epoch * config.OPTIMIZER_CONFIG['lr'] / self.batch_size)
        return np.array(grad)

    def set_gradient(self, gradient):
        difference = np.subtract(self.get_gradient(), gradient)
        print(np.linalg.norm(difference))
        new_parameter = np.subtract(self.global_current.flatten_model(), gradient)
        new_parameter = self.client_current.unflatten_model(new_parameter)

        self.client_current.load_state_dict(new_parameter)


    def get_gradient_s(self, model1, model2, difference=False):
        grad = np.subtract(model1.flatten_model(), model2.flatten_model())
        if difference:
            return grad
        # return grad / (args.num_sample * args.local_ep * lr / args.local_bs)
        grad = grad / (len(self.train) * self.local_epoch * config.OPTIMIZER_CONFIG['lr'] / self.batch_size)
        return grad

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.train)

    def mutate(self):
        self.drift = True

    def update_train(self, new_dataset, replace=False):
        if self.train is None or replace:
            self.train = new_dataset
        else:
            self.train + new_dataset

    def update_test(self, new_dataset, replace=True):
        if self.test is None or replace:
            self.test = new_dataset
        else:
            self.test + new_dataset

    def client_update(self, run_type):
        if run_type == 'fedprox':
            return self.client_update_fedprox()
        elif run_type == 'scaffold':
            return self.client_update_scaffold()
        else:
            return self.client_update_fedavg()

    def client_update_fedavg(self):
        """Update local model using local dataset."""
        self.client_current.train()
        self.client_current.to(self.device)

        optimizer = eval(self.optimizer)(self.client_current.parameters(), **self.optim_config)
        for e in range(self.local_epoch):
            for data, labels in self.train.get_dataloader():
                data, labels = data.float().to(self.device), labels.long().to(self.device)

                optimizer.zero_grad()
                outputs = self.client_current(data)
                loss = eval(self.criterion)()(outputs, labels)

                loss.backward()
                optimizer.step()

                if self.device == "cuda": torch.cuda.empty_cache()

        self.client_current.to("cpu")

    def client_update_fedprox(self):
        """Update local model using local dataset."""
        self.client_current.train()
        self.client_current.to(self.device)

        global_weight_collector = list(self.global_current.to(self.device).parameters())
        mu = 0.001

        optimizer = eval(self.optimizer)(self.client_current.parameters(), **self.optim_config)
        for e in range(self.local_epoch):
            for data, labels in self.train.get_dataloader():
                data, labels = data.float().to(self.device), labels.long().to(self.device)

                optimizer.zero_grad()
                outputs = self.client_current(data)
                loss = eval(self.criterion)()(outputs, labels)

                fed_prox_reg = 0.0
                for param_index, param in enumerate(self.client_current.parameters()):
                    fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                loss += fed_prox_reg

                loss.backward()
                optimizer.step()

                if self.device == "cuda": torch.cuda.empty_cache()

        self.client_current.to("cpu")

    def client_update_scaffold(self):
        """Update local model using local dataset."""
        if self.c_global is None:
            self.c_global = copy.deepcopy(self.client_current)
        if self.c_local is None:
            self.c_local = copy.deepcopy(self.client_current)

        self.client_current.train()
        self.client_current.to(self.device)
        self.c_global.to(self.device)
        self.c_local.to(self.device)
        self.global_current.to(self.device)

        c_global_para = self.c_global.state_dict()
        c_local_para = self.c_local.state_dict()
        count = 0
        optimizer = eval(self.optimizer)(self.client_current.parameters(), **self.optim_config)
        for e in range(self.local_epoch):
            for data, labels in self.train.get_dataloader():
                data, labels = data.float().to(self.device), labels.long().to(self.device)

                optimizer.zero_grad()
                outputs = self.client_current(data)
                loss = eval(self.criterion)()(outputs, labels)

                loss.backward()
                optimizer.step()

                net_para = self.client_current.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key] - config.OPTIMIZER_CONFIG['lr'] * \
                                               (c_global_para[key] - c_local_para[key])
                self.client_current.load_state_dict(net_para)
                count += 1

                if self.device == "cuda": torch.cuda.empty_cache()

        self.client_current.to("cpu")
        self.c_global.to("cpu")
        self.c_local.to("cpu")
        self.global_current.to("cpu")

        c_new_para = self.c_local.state_dict()
        c_delta_para = copy.deepcopy(self.c_local.state_dict())
        global_current_para = self.global_current.state_dict()
        client_current_para = self.client_current.state_dict()

        c_global_para = self.c_global.state_dict()
        c_local_para = self.c_local.state_dict()

        for key in client_current_para:
            c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_current_para[key] - client_current_para[key]) / \
                              (count * config.OPTIMIZER_CONFIG['lr'])

            c_delta_para[key] = c_new_para[key] - c_local_para[key]
        self.c_local.load_state_dict(c_new_para)
        self.c_delta_para = c_delta_para
        # print(self.c_delta_para)

    def evaluate(self, model, dataset):
        model.eval()
        model.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in dataset.get_dataloader():
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = model(data)
                test_loss += eval(self.criterion)()(outputs, labels).item()

                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                if self.device == "cuda": torch.cuda.empty_cache()
        model.to("cpu")

        test_loss = test_loss / len(dataset.get_dataloader())
        test_accuracy = correct / len(dataset)

        return test_accuracy, test_loss

    def client_evaluate(self):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        self.client_current.eval()
        self.client_current.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in self.test.get_dataloader():
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                test_loss += eval(self.criterion)()(outputs, labels).item()

                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                if self.device == "cuda": torch.cuda.empty_cache()
        self.client_current.to("cpu")

        test_loss = test_loss / len(self.test.get_dataloader())
        test_accuracy = correct / len(self.test)

        message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation!\
            \n\t=> Test loss: {test_loss:.4f}\
            \n\t=> Test accuracy: {100. * test_accuracy:.2f}%\
            \n\t=> Distribution: {self.distribution}\n"

        print(message, flush=True);
        logging.info(message)

        del message;
        gc.collect()

        return test_loss, test_accuracy

    def get_performance_gap(self):
        _, global_accuracy = self.client_evaluate(current_model=False, log=False)
        _, current_accuracy = self.client_evaluate(current_model=True, log=False)

        self.idx_t1 = current_accuracy - global_accuracy

        # print(id(self.model))
        # print(id(self.global_model))

        if self.idx_t0 is None:
            self.idx_t0 = self.idx_t1
        else:
            res = self.idx_t1 - self.idx_t0

            message = f"\t[Client {str(self.id).zfill(4)}]:!\
                \n\t=> Global Accuracy: {100. * global_accuracy:.2f}%\
                \n\t=> Current Accuracy: {100. * current_accuracy:.2f}%\
                \n\t=> idx_t0: {100. * self.idx_t0: .2f}%\
                \n\t=> idx_t1: {100. * self.idx_t1: .2f}%"

            print(message, flush=True);
            logging.info(message)

            return max(res, 0.000001)
