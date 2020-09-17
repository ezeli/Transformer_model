# coding:utf8
import math
import torch
from torch import nn
from collections import namedtuple


BeamCandidate = namedtuple('BeamCandidate',
                           ['log_prob_sum', 'log_prob_seq', 'word_id_seq'])


class MultiHeadAttention(nn.Module):
    def __init__(self, settings):
        super(MultiHeadAttention, self).__init__()
        assert settings['d_model'] % settings['h'] == 0
        self.h = settings['h']
        self.d_k = settings['d_model'] // settings['h']
        self.linears = nn.ModuleList([nn.Linear(settings['d_model'], settings['d_model']) for _ in range(4)])
        self.drop = nn.Dropout(settings['dropout_p'])

    def _attention(self, query, key, value, mask=None):
        # Scaled Dot Product Attention
        scores = query.matmul(key.transpose(-2, -1)) \
                 / math.sqrt(self.d_k)  # bs*h*n1*n2
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        scores = scores.softmax(-1)
        return scores.matmul(value)  # bs*h*n1*d_k

    def forward(self, query, key, value, mask=None):
        """
            query: bs*n1*d_model
            key: bs*n2*d_model
            value: bs*n2*d_model
            mask: bs*(n2 or 1)*n2
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # bs*1*(n2 or 1)*n2
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [self.drop(l(x)).reshape(batch_size, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears[:3], (query, key, value))]  # bs*h*n*d_k

        # 2) Apply attention on all the projected vectors in batch.
        x = self._attention(query, key, value, mask)  # bs*h*n1*d_k

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).reshape(batch_size, -1, self.h * self.d_k)  # bs*n1*d_model
        return self.drop(self.linears[-1](x))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, settings):
        super(PositionwiseFeedForward, self).__init__()
        self.pff = nn.Sequential(
            nn.Linear(settings['d_model'], settings['d_ff']),
            nn.ReLU(),
            # nn.Dropout(settings['dropout_p']),
            nn.Linear(settings['d_ff'], settings['d_model'])
        )

    def forward(self, x):
        return self.pff(x)


class PositionalEncoding(nn.Module):
    def __init__(self, settings):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(settings['max_seq_len'], settings['d_model'])
        position = torch.arange(0, settings['max_seq_len']).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, settings['d_model'], 2).float() *
                             -(math.log(10000.0) / settings['d_model']))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.drop = nn.Dropout(settings['dropout_p'])

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.drop(x)


class EncoderLayer(nn.Module):
    def __init__(self, settings):
        super(EncoderLayer, self).__init__()
        self.multi_head_att = MultiHeadAttention(settings)
        self.feed_forward = PositionwiseFeedForward(settings)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(settings['d_model']) for _ in range(2)])
        self.drop = nn.Dropout(settings['dropout_p'])

    def _add_res_connection(self, x, sublayer, n):
        return self.layer_norms[n](x + self.drop(sublayer(x)))  # x + self.drop(sublayer(self.layer_norms[n](x)))

    def forward(self, x, mask):
        x = self._add_res_connection(x, lambda x: self.multi_head_att(x, x, x, mask), 0)
        return self._add_res_connection(x, self.feed_forward, 1)


class Encoder(nn.Module):
    def __init__(self, settings):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(settings) for _ in range(settings['N_enc'])])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, settings):
        super(DecoderLayer, self).__init__()
        self.masked_multi_head_att = MultiHeadAttention(settings)
        self.multi_head_att = MultiHeadAttention(settings)
        self.feed_forward = PositionwiseFeedForward(settings)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(settings['d_model']) for _ in range(3)])
        self.drop = nn.Dropout(settings['dropout_p'])

    def _add_res_connection(self, x, sublayer, n):
        return self.layer_norms[n](x + self.drop(sublayer(x)))  # x + self.drop(sublayer(self.layer_norms[n](x)))

    def forward(self, captions, seq_masks, enc_out, att_masks):
        x = self._add_res_connection(captions, lambda x: self.masked_multi_head_att(x, x, x, seq_masks), 0)
        x = self._add_res_connection(x, lambda x: self.multi_head_att(x, enc_out, enc_out, att_masks), 1)
        return self._add_res_connection(x, self.feed_forward, 2)


class Decoder(nn.Module):
    def __init__(self, settings):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(settings) for _ in range(settings['N_dec'])])

    def forward(self, captions, seq_masks, enc_out, att_masks):
        for layer in self.layers:
            captions = layer(captions, seq_masks, enc_out, att_masks)
        return captions


class Transformer(nn.Module):
    def __init__(self, idx2word, settings):
        super(Transformer, self).__init__()
        self.idx2word = idx2word
        self.pad_id = idx2word.index('<PAD>')
        self.unk_id = idx2word.index('<UNK>')
        self.sos_id = idx2word.index('<SOS>') if '<SOS>' in idx2word else self.pad_id
        self.eos_id = idx2word.index('<EOS>') if '<EOS>' in idx2word else self.pad_id
        self.max_seq_len = settings['max_seq_len']

        self.d_model = settings['d_model']
        self.vocab_size = len(idx2word)
        self.word_embed = nn.Embedding(self.vocab_size, settings['d_model'], padding_idx=self.pad_id)
        self.pe = PositionalEncoding(settings)
        self.att_embed = nn.Sequential(nn.Linear(settings['att_feat_dim'], settings['d_model']),
                                       nn.ReLU(),
                                       nn.Dropout(settings['dropout_p']))

        self.encoder = Encoder(settings)
        self.decoder = Decoder(settings)

        self.classifier = nn.Linear(settings['d_model'], self.vocab_size)

    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'xe')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, 'forward_' + mode)(*args, **kwargs)

    def _feats_encode(self, att_feats, att_masks=None):
        att_feats = att_feats.reshape(att_feats.size(0), -1, att_feats.size(-1))  # bs*num_atts*feat_emb
        att_feats = self.att_embed(att_feats)  # bs*num_atts*d_model

        if att_masks is not None:
            att_masks = att_masks.unsqueeze(-2)  # bs*1*num_atts
        return att_feats, att_masks

    def _sequence_encode(self, captions, lengths=None):
        seq_len = captions.size(-1)
        captions = self.word_embed(captions) * math.sqrt(self.d_model)  # bs*seq_len*d_model
        captions = self.pe(captions)

        if lengths is None:
            seq_masks = captions.new_ones(1, seq_len, seq_len, dtype=torch.bool).tril(diagonal=0)  # 1*seq_len*seq_len
        else:
            assert seq_len == max(lengths)
            seq_masks = captions.new_zeros(len(lengths), seq_len, dtype=torch.bool)  # bs*seq_len
            for i, l in enumerate(lengths):
                seq_masks[i, :l] = True
            seq_masks = seq_masks.unsqueeze(-2)  # bs*1*seq_len
            seq_masks = seq_masks & seq_masks.new_ones(1, seq_len, seq_len).tril(diagonal=0)  # bs*seq_len*seq_len
        return captions, seq_masks

    def _encode(self, att_feats, att_masks=None):
        att_feats, att_masks = self._feats_encode(att_feats, att_masks)
        enc_out = self.encoder(att_feats, att_masks)  # bs*num_atts*d_model
        return enc_out, att_masks

    def _decode(self, captions, lengths, enc_out, att_masks):
        captions, seq_masks = self._sequence_encode(captions, lengths)  # bs*seq_len*d_model, bs*seq_len*seq_len
        dec_out = self.decoder(captions, seq_masks, enc_out, att_masks)  # bs*seq_len*d_model
        dec_out = self.classifier(dec_out).log_softmax(dim=-1)  # bs*seq_len*vocab
        return dec_out

    def forward_xe(self, att_feats, captions, lengths):
        enc_out, att_masks = self._encode(att_feats)  # bs*num_atts*d_model, bs*1*num_atts
        dec_out = self._decode(captions[:, :-1], lengths, enc_out, att_masks)
        return dec_out

    def forward_rl(self, att_feats, sample_max):
        batch_size = att_feats.size(0)
        enc_out, att_masks = self._encode(att_feats)  # bs*num_atts*d_model, bs*1*num_atts

        seq = att_feats.new_zeros((batch_size, self.max_seq_len), dtype=torch.long)
        seq_logprobs = att_feats.new_zeros((batch_size, self.max_seq_len))
        seq_masks = att_feats.new_zeros((batch_size, self.max_seq_len))
        it = att_feats.new_zeros(batch_size, dtype=torch.long).fill_(self.sos_id)  # first input <SOS>
        unfinished = it == self.sos_id
        pre_words = it.unsqueeze(1)  # bs*1
        for t in range(self.max_seq_len):
            logprobs = self._decode(pre_words, None, enc_out, att_masks)  # bs*seq_len*vocab
            logprobs = logprobs[:, -1]  # bs*vocab

            if sample_max:
                sample_logprobs, it = torch.max(logprobs, 1)
            else:
                prob_prev = torch.exp(logprobs)
                it = torch.multinomial(prob_prev, 1)
                sample_logprobs = logprobs.gather(1, it)  # gather the logprobs at sampled positions
            it = it.view(-1).long()
            sample_logprobs = sample_logprobs.view(-1)

            seq_masks[:, t] = unfinished
            it = it * unfinished.type_as(it)  # bs
            seq[:, t] = it
            seq_logprobs[:, t] = sample_logprobs
            pre_words = torch.cat([pre_words, it.unsqueeze(1)], dim=1)  # bs*seq_len

            unfinished = unfinished * (it != self.eos_id)
            if unfinished.sum() == 0:
                break

        return seq, seq_logprobs, seq_masks

    def sample(self, att_feat, beam_size=3, decoding_constraint=1):
        self.eval()
        att_feat = att_feat.view(1, -1, att_feat.size(-1))  # 1*num_atts*2048
        enc_out, att_masks = self._encode(att_feat)  # 1*num_atts*d_model, 1*1*num_atts

        # log_prob_sum, log_prob_seq, word_id_seq
        candidates = [BeamCandidate(0., [], [self.sos_id])]
        for t in range(self.max_seq_len):
            tmp_candidates = []
            end_flag = True
            for candidate in candidates:
                log_prob_sum, log_prob_seq, word_id_seq = candidate
                if t > 0 and word_id_seq[-1] == self.eos_id:
                    tmp_candidates.append(candidate)
                else:
                    end_flag = False
                    pre_words = att_feat.new_tensor(word_id_seq, dtype=torch.long).unsqueeze(0)  # 1*seq_len
                    logprobs = self._decode(pre_words, None, enc_out, att_masks)  # 1*seq_len*vocab
                    logprobs = logprobs[:, -1]  # 1*vocab
                    logprobs = logprobs.squeeze(0)  # vocab_size
                    if self.pad_id != self.eos_id:
                        logprobs[self.pad_id] += float('-inf')  # do not generate <PAD>, <SOS> and <UNK>
                        logprobs[self.sos_id] += float('-inf')
                        logprobs[self.unk_id] += float('-inf')
                    if decoding_constraint:  # do not generate last step word
                        logprobs[word_id_seq[-1]] += float('-inf')

                    output_sorted, index_sorted = torch.sort(logprobs, descending=True)
                    for k in range(beam_size):
                        log_prob, word_id = output_sorted[k], index_sorted[k]  # tensor, tensor
                        log_prob = float(log_prob)
                        word_id = int(word_id)
                        tmp_candidates.append(BeamCandidate(log_prob_sum + log_prob,
                                                            log_prob_seq + [log_prob],
                                                            word_id_seq + [word_id]))
            candidates = sorted(tmp_candidates, key=lambda x: x.log_prob_sum, reverse=True)[:beam_size]
            if end_flag:
                break

        # captions, scores
        captions = [' '.join([self.idx2word[idx] for idx in candidate.word_id_seq[1:-1]])
                    for candidate in candidates]
        scores = [candidate.log_prob_sum for candidate in candidates]
        return captions, scores

    def get_optim_and_crit(self, lr, weight_decay=0, smoothing=0.1):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay), \
               LabelSmoothingCriterion(smoothing)


class XECriterion(nn.Module):
    def __init__(self):
        super(XECriterion, self).__init__()

    def forward(self, pred, target, lengths):
        max_len = max(lengths)
        mask = pred.new_zeros(len(lengths), max_len)
        for i, l in enumerate(lengths):
            mask[i, :l] = 1

        loss = - pred.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        loss = torch.sum(loss) / torch.sum(mask)

        return loss


class LabelSmoothingCriterion(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCriterion, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.true_dist = None

    def forward(self, pred, target, lengths):
        max_len = max(lengths)
        mask = pred.new_zeros(len(lengths), max_len)
        for i, l in enumerate(lengths):
            mask[i, :l] = 1

        vocab_size = pred.size(-1)
        pred = pred.reshape(-1, vocab_size)  # [bs*seq_len, vocab]
        target = target.reshape(-1)  # [bs*seq_len]
        mask = mask.reshape(-1)  # [bs*seq_len]

        true_dist = pred.clone()
        true_dist.fill_(self.smoothing / (vocab_size - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        loss = (self.criterion(pred, true_dist.detach()).sum(1) * mask).sum() / mask.sum()
        return loss
