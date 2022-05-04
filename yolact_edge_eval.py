import logging
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from eval import parse_args, Detections, prep_display
from yolact_edge.data import set_cfg
from yolact_edge.utils import timer
from yolact_edge.utils.augmentations import FastBaseTransform
from yolact_edge.utils.functions import SavePath, MovingAverage, ProgressBar
from yolact_edge.utils.logging_helper import setup_logger
from yolact_edge.yolact import Yolact

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes


class YolactPredict(object):
    def __init__(self, trained_model):
        self.cfg = None
        self.args = None
        self.net = self.init_net(trained_model)

    def init_net(self, trained_model):
        self.args = parse_args()

        # init config
        model_path = SavePath.from_str(trained_model)
        self.args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % self.args.config)
        self.cfg = set_cfg(self.args.config)

        # init logger
        setup_logger(logging_level=logging.INFO)
        logger = logging.getLogger("yolact.eval")

        with torch.no_grad():
            if self.args.cuda:
                cudnn.benchmark = True
                cudnn.fastest = True
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            else:
                torch.set_default_tensor_type('torch.FloatTensor')

            # init model
            logger.info('Loading model...')
            net = Yolact(training=False)
            net.load_weights(self.args.trained_model, args=self.args)
            net.eval()
            logger.info('Model loaded.')
            if self.args.cuda:
                net = net.cuda()

        return net

    def save_video(self, in_path: str, out_path: str):
        vid = cv2.VideoCapture(in_path)

        target_fps = round(vid.get(cv2.CAP_PROP_FPS))
        frame_width = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = round(vid.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (frame_width, frame_height))

        transform = FastBaseTransform()
        frame_times = MovingAverage()
        progress_bar = ProgressBar(30, num_frames)

        every_k_frames = 5
        moving_statistics = {"conf_hist": []}

        try:
            for i in range(num_frames):
                timer.reset()
                frame_idx = i
                with timer.env('Video'):
                    frame = torch.from_numpy(vid.read()[1]).cuda().float()
                    batch = transform(frame.unsqueeze(0))
                    # preds = net(batch)

                    if frame_idx % every_k_frames == 0 or self.cfg.flow.warp_mode == 'none':
                        extras = {"backbone": "full", "interrupt": False, "keep_statistics": True,
                                  "moving_statistics": moving_statistics}

                        with torch.no_grad():
                            net_outs = self.net(batch, extras=extras)

                        moving_statistics["feats"] = net_outs["feats"]
                        moving_statistics["lateral"] = net_outs["lateral"]

                    else:
                        extras = {"backbone": "partial", "interrupt": False, "keep_statistics": False,
                                  "moving_statistics": moving_statistics}

                        with torch.no_grad():
                            net_outs = self.net(batch, extras=extras)

                    preds = net_outs["pred_outs"]

                    processed = prep_display(preds, frame, None, None, undo_transform=False, class_color=True)

                    out.write(processed)

                if i > 1:
                    frame_times.add(timer.total_time())
                    fps = 1 / frame_times.get_avg()
                    progress = (i + 1) / num_frames * 100
                    progress_bar.set_val(i + 1)

                    print('\rProcessing Frames  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                          % (repr(progress_bar), i + 1, num_frames, progress, fps), end='')

        except KeyboardInterrupt:
            print('Stopping early.')

        vid.release()
        out.release()
        print()

    def eval_image(self, path: str, save_path: str = None, detections: Detections = None, image_id=None):
        frame = torch.from_numpy(cv2.imread(path)).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        extras = {"backbone": "full", "interrupt": False, "keep_statistics": False, "moving_statistics": None}
        preds = self.net(batch, extras=extras)["pred_outs"]
        img_numpy = prep_display(preds, frame, None, None, undo_transform=False)
        cv2.imwrite(save_path, img_numpy)

    def evaluate(self, source_path):
        self.net.detect.use_fast_nms = self.args.fast_nms
        self.cfg.mask_proto_debug = self.args.mask_proto_debug

        source_path = Path(source_path)
        source_path_suffix = source_path.suffix.replace(".", '')

        if source_path_suffix in IMG_FORMATS:
            source_type = 'image'
        elif source_path_suffix in VID_FORMATS:
            source_type = 'video'
        else:
            print("Source type error")
            return -1

        source_path_out = source_path.with_name(source_path.stem + '_out' + source_path.suffix)

        with torch.no_grad():
            if source_type == 'image':
                self.eval_image(str(source_path), str(source_path_out), image_id="0")

            elif source_type == 'video':
                self.save_video(str(source_path), str(source_path_out))

            else:
                print("Source type error")

        print(f'eval done, saved in {str(source_path_out)}')


if __name__ == '__main__':
    yolact = YolactPredict(
        trained_model=r'E:\AI_Project\yolact_edge\weights\repo_best_weights\yolact_edge_resnet50_54_800000.pth')
    video_path = r'E:\AI_Project\yolact_edge\data\test_video.mp4'
    image_path = r'E:\AI_Project\yolact_edge\data\test\000000000009.jpg'
    yolact.evaluate(video_path)
    yolact.evaluate(image_path)
