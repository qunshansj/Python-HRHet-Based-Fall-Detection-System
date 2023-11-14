
python scripts/train_coco.py --help
usage: train_coco.py [-h] [--exp_name EXP_NAME] [--epochs EPOCHS]
                     [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS]
                     [--lr LR] [--disable_lr_decay]
                     [--lr_decay_steps LR_DECAY_STEPS]
                     [--lr_decay_gamma LR_DECAY_GAMMA] [--optimizer OPTIMIZER]
                     [--weight_decay WEIGHT_DECAY] [--momentum MOMENTUM]
                     [--nesterov]
                     [--pretrained_weight_path PRETRAINED_WEIGHT_PATH]
                     [--checkpoint_path CHECKPOINT_PATH] [--log_path LOG_PATH]
                     [--disable_tensorboard_log] [--model_c MODEL_C]
                     [--model_nof_joints MODEL_NOF_JOINTS]
                     [--model_bn_momentum MODEL_BN_MOMENTUM]
                     [--disable_flip_test_images]
                     [--image_resolution IMAGE_RESOLUTION]
                     [--coco_root_path COCO_ROOT_PATH]
                     [--coco_bbox_path COCO_BBOX_PATH] [--seed SEED]
                     [--device DEVICE]

optional arguments:
  -h, --help            show this help message and exit
  --exp_name EXP_NAME, -n EXP_NAME
                        experiment name. A folder with this name will be
                        created in the log_path.
  --epochs EPOCHS, -e EPOCHS
                        number of epochs
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        batch size
  --num_workers NUM_WORKERS, -w NUM_WORKERS
                        number of DataLoader workers
  --lr LR, -l LR        initial learning rate
  --disable_lr_decay    disable learning rate decay
  --lr_decay_steps LR_DECAY_STEPS
                        learning rate decay steps
  --lr_decay_gamma LR_DECAY_GAMMA
                        learning rate decay gamma
  --optimizer OPTIMIZER, -o OPTIMIZER
                        optimizer name. Currently, only `SGD` and `Adam` are
                        supported.
  --weight_decay WEIGHT_DECAY
                        weight decay
  --momentum MOMENTUM, -m MOMENTUM
                        momentum
  --nesterov            enable nesterov
  --pretrained_weight_path PRETRAINED_WEIGHT_PATH, -p PRETRAINED_WEIGHT_PATH
                        pre-trained weight path. Weights will be loaded before
                        training starts.
  --checkpoint_path CHECKPOINT_PATH, -c CHECKPOINT_PATH
                        previous checkpoint path. Checkpoint will be loaded
                        before training starts. It includes the model, the
                        optimizer, the epoch, and other parameters.
  --log_path LOG_PATH   log path. tensorboard logs and checkpoints will be
                        saved here.
  --disable_tensorboard_log, -u
                        disable tensorboard logging
  --model_c MODEL_C     HRNet c parameter
  --model_nof_joints MODEL_NOF_JOINTS
                        HRNet nof_joints parameter
  --model_bn_momentum MODEL_BN_MOMENTUM
                        HRNet bn_momentum parameter
  --disable_flip_test_images
                        disable image flip during evaluation
  --image_resolution IMAGE_RESOLUTION, -r IMAGE_RESOLUTION
                        image resolution
  --coco_root_path COCO_ROOT_PATH
                        COCO dataset root path
  --coco_bbox_path COCO_BBOX_PATH
                        path of detected bboxes to use during evaluation
  --seed SEED, -s SEED  seed
  --device DEVICE, -d DEVICE
                        device
