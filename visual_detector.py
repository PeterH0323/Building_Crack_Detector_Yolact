import logging

import torch
import torch.backends.cudnn as cudnn

from eval import evaluate, parse_args
from yolact_edge.data import set_cfg
from yolact_edge.utils.functions import SavePath
from yolact_edge.yolact import Yolact


def init_net():
    args = parse_args()

    # init config
    model_path = SavePath.from_str(args.trained_model)
    # TODO: Bad practice? Probably want to do a name lookup instead.
    args.config = model_path.model_name + '_config'
    print('Config not specified. Parsed %s from the file name.\n' % args.config)
    set_cfg(args.config)

    # init logger
    from yolact_edge.utils.logging_helper import setup_logger
    setup_logger(logging_level=logging.INFO)
    logger = logging.getLogger("yolact.eval")

    with torch.no_grad():
        if args.cuda:
            cudnn.benchmark = True
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        # init dataset
        dataset = None

        # init model
        logger.info('Loading model...')
        net = Yolact(training=False)
        net.load_weights(args.trained_model, args=args)
        net.eval()
        logger.info('Model loaded.')
        if args.cuda:
            net = net.cuda()
        evaluate(net, dataset)


if __name__ == '__main__':
    init_net()
