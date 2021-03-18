# coding:utf8
import torch
import skimage.io

from opts import parse_opt
from models.decoder import Transformer
from models.encoder import Encoder

opt = parse_opt()
assert opt.test_model, 'please input test_model'
assert opt.image_file, 'please input image_file'

encoder = Encoder(opt.resnet101_file)
encoder.to(opt.device)
encoder.eval()

img = skimage.io.imread(opt.image_file)
with torch.no_grad():
    img = encoder.preprocess(img)
    img = img.to(opt.device)
    _, att_feat = encoder(img)

print("====> loading checkpoint '{}'".format(opt.test_model))
chkpoint = torch.load(opt.test_model, map_location=lambda s, l: s)
model = Transformer(chkpoint['idx2word'], chkpoint['settings'])
model.load_state_dict(chkpoint['model'])
print("====> loaded checkpoint '{}', epoch: {}, train_mode: {}".
      format(opt.test_model, chkpoint['epoch'], chkpoint['train_mode']))
model.to(opt.device)
model.eval()

rest, _ = model.sample(att_feat, beam_size=opt.beam_size)
print('generate captions:\n' + '\n'.join(rest))
