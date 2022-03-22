import argparse
import os
import torch
import logging
import numpy as np
from model import get_model
from model.optimizer import init_optimizer

from tools.init_tool import init_all
from config_parser import create_config
from tools.train_tool import train

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="specific config file", required=True)
    parser.add_argument('--gpu', '-g', help="gpu id list")
    parser.add_argument('--checkpoint', help="checkpoint file path")
    args = parser.parse_args()

    configFilePath = args.config

    use_gpu = True
    gpu_list = []
    if args.gpu is None:
        use_gpu = False
    else:
        use_gpu = True
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        device_list = args.gpu.split(",")
        for a in range(0, len(device_list)):
            gpu_list.append(int(a))

    os.system("clear")

    config = create_config(configFilePath)

    cuda = torch.cuda.is_available()
    logger.info("CUDA available: %s" % str(cuda))
    if not cuda and len(gpu_list) > 0:
        logger.error("CUDA is not available but specific gpu id")
        raise NotImplementedError

    parameters = init_all(config, gpu_list, args.checkpoint, "train")
    if config.get('model', 'grid') == "True":
        file_path = config.get('model', 'file_path')
        for i in range(1, 11):
            config.set('model', 'label_weight', i)
            p, r, f1, epochs = [], [], [], []
            for j in range(5):
                result, epoch = train(parameters, config, gpu_list)
                parameters['model'] = get_model(config.get("model", "model_name"))(config, gpu_list, args.checkpoint, "train")
                if len(gpu_list) > 0:
                    parameters['model'] = parameters['model'].cuda()
                    parameters['model'].init_multi_gpu(gpu_list, config, args.checkpoint, "train")
                parameters['optimizer'] = init_optimizer(parameters['model'], config, args.checkpoint, "train")
                p.append(result['precision'])
                r.append(result['recall'])
                f1.append(result['f1'])
                epochs.append(epoch)
                with open(file_path, 'a') as f:
                    f.write(f'p = {result["precision"]}, r = {result["recall"]}, f1 = {result["f1"]}, epoch = {epoch}\n')
            with open(file_path, 'a') as f:
                f.write(f'label_weight={i}, p = {np.mean(p)}±{np.std(p)}, r = {np.mean(r)}±{np.std(r)}, f1 = {np.mean(f1)}±{np.std(f1)}, epoch = {np.mean(epochs)}±{np.std(epochs)}\n')
    else:
        train(parameters, config, gpu_list)