import gc
import os
import datetime
from torch.utils.tensorboard import SummaryWriter
import logging

# custom packages
from ..config import config


class Printer:
    def __init__(self):
        # modify log_path to contain current time
        log_path = os.path.join(config.LOG_PATH, config.RUN_NAME + '_' + str(datetime.datetime.now().strftime("%Y-%m-%d")))

        # initiate TensorBoard for tracking losses and metrics
        self.writer = SummaryWriter(log_dir=log_path, filename_suffix="FL")
        # os.system(f"tensorboard --logdir={log_path} --port={config.TB_PORT} --host={config.TB_HOST}")

        logger = logging.getLogger(__name__)
        logging.basicConfig(
            filename=os.path.join(log_path, config.LOG_NAME),
            level=logging.INFO,
            format="[%(levelname)s](%(asctime)s) %(message)s",
            datefmt="%Y/%m/%d/ %I:%M:%S %p")

    def print(self, message):
        print(message); logging.info(message)
        del message; gc.collect()

    def get_tensorboard_writer(self):
        return self.writer

def pretty_list(list):
    text = str(["{:.2f}".format(i) for i in list])
    return text