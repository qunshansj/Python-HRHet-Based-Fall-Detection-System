
def calCenterScale(self, bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    center_w = (bbox[0] + bbox[2]) / 2.0
    center_h = (bbox[1] + bbox[3]) / 2.0
    scale = round((max(w, h) / 200.0), 2)
    return center_w, center_h, scale
    
def __getitem__(self, idx):
    image_path = os.path.join(self.data_root,
                                  self.landmarks_frame[idx]["image_path"])
    bbox = self.landmarks_frame[idx]['bbox']
    center_w, center_h, scale = self.calCenterScale(bbox)
    center = torch.Tensor([center_w, center_h])
    pts = np.array(self.landmarks_frame[idx]["keypoints"])
    pts = pts.astype('float').reshape(-1, 2)       
    ...
