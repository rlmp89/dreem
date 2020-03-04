import argparse
import torch
from tqdm import tqdm
import data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.arch as module_arch
from parse_config import ConfigParser
import numpy as np

def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=config['data_loader']['args']['batch_size'],
        shuffle=False,
        validation_split=0.0,
        training=False,
        testing=False,
        num_workers=config['data_loader']['args']['num_workers']
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

 
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        pred_all=[]
        for i, data in enumerate(tqdm(data_loader)):
            data = data.to(device)
            output = model(data)
            pred_all.append(output)
        pred_all = torch.cat(pred_all)
        y_pred_merged = torch.stack(pred_all.chunk(40),axis=1).mean(axis=1).argmax(axis=1)   
        pred = y_pred_merged.cpu().numpy()
        np.savetxt(config["output"],pred,fmt='%d')
   


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-o', '--output', default=None, type=str,
                      help='output file (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    config = ConfigParser.from_args(args)
    main(config)
