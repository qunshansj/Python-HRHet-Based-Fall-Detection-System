
DATASET:
  DATASET: OWN
  ROOT: '../data/own/images'
  TRAINSET: '../data/own/train.json'
  TESTSET: '../data/own/val.json'
  FLIP: true
  SCALE_FACTOR: 0.25
  ROT_FACTOR: 30
MODEL:
  NAME: 'hrnet'
  NUM_JOINTS: 37  // 根据自己数据集特征点数量
  INIT_WEIGHTS: true
  PRETRAINED: 'hrnetv2_pretrained/hrnetv2_w18_imagenet_pretrained.pth'
