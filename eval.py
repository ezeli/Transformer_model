# coding:utf8
import torch
import json
import tqdm

from opts import parse_opt
from models.decoder import Transformer
from dataloader import get_dataloader

opt = parse_opt()
assert opt.eval_model, 'please input eval_model'
assert opt.result_file, 'please input result_file'

print("====> loading checkpoint '{}'".format(opt.eval_model))
chkpoint = torch.load(opt.eval_model, map_location=lambda s, l: s)
model = Transformer(chkpoint['idx2word'], chkpoint['settings'])
model.load_state_dict(chkpoint['model'])
print("====> loaded checkpoint '{}', epoch: {}, train_mode: {}".
      format(opt.eval_model, chkpoint['epoch'], chkpoint['train_mode']))
model.to(opt.device)
model.eval()

captions = json.load(open(opt.captions, 'r'))
test_captions = {}
for fn in captions['test']:
    test_captions[fn] = [[]]
test_data = get_dataloader(opt.att_feats, test_captions, model.pad_id,
                           model.max_seq_len, opt.batch_size, opt.num_workers, shuffle=False)

results = []
for fns, att_feats, _, _ in tqdm.tqdm(test_data, ncols=100):
    att_feats = att_feats.to(opt.device)
    for i, fn in enumerate(fns):
        att_feat = att_feats[i]
        with torch.no_grad():
            rest, _ = model.sample(att_feat, beam_size=opt.beam_size)
        results.append({'image_id': fn, 'caption': rest[0]})
json.dump(results, open(opt.result_file, 'w'))
