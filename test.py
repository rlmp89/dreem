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
        testing=True,
        num_workers=config['data_loader']['args']['num_workers']
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss']['type'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    #some metrics cannot be reduced across batches (ex roc_auc)
    batch_metric_fns = [m for m in metric_fns if not hasattr(m, 'global_metric')]
    global_metric_fns = [m for m in metric_fns if hasattr(m, 'global_metric')]
    need_all_outputs= len(global_metric_fns)> 0


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

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        pred_all=[]
        target_all =[]
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            if need_all_outputs:
                pred_all.append(output)
                target_all.append(target)


            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for j, metric in enumerate(batch_metric_fns):
                total_metrics[j] += metric(output, target) * batch_size


    n_samples = len(data_loader.sampler)
   
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(batch_metric_fns)
    })

    if need_all_outputs:
            with torch.no_grad():
                pred_all = torch.cat(pred_all)
                target_all = torch.cat(target_all)
                for metric in global_metric_fns:
                    log.update({
                            metric.__name__: metric(pred_all, target_all) 
                        }) 
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
