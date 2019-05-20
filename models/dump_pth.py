import torch
from mobilenet import MBv2
model_path = 'mbv2x1_imagenet.pth.tar'
ckpt = torch.load(model_path)
for k,v in ckpt.items():
    print (k,v)

model = MBv2(num_classes)
ckpt = torch.load(url)
# original saved file with DataParallel
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in ckpt['state_dict'].items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
