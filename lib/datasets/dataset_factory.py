from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from .sample.ctdet import CTDetDataset


from .dataset.coco import COCO
from .dataset.pascal import PascalVOC
from .dataset.kitti import KITTI
from .dataset.coco_hp import COCOHP
from .dataset.shipcoco import shipcoco


dataset_factory = {
  'coco': COCO,
  'pascal': PascalVOC,
  'kitti': KITTI,
  'coco_hp': COCOHP,
  'shipcoco':shipcoco
}

_sample_factory = {
  'ctdet': CTDetDataset,
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
def collate_fn(self, batch):
    ret, targets = list(zip(*batch))

    targets = [boxes for boxes in targets if boxes is not None]

    targets = torch.cat(targets, 0)

    return ret, targets