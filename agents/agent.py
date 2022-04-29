# -*- coding: utf-8 -*-
from locale import normalize
import numpy as np
import io
import sys
import soundfile as sf
import uuid
from NMTGMinor.onmt.utils import safe_readaudio
from fairseq.models.wav2vec import Wav2Vec2Config
import os
import torch
import torchaudio
sys.path.insert(0, '/home/ppolak/iwslt')
sys.path.insert(0, '/home/ppolak/iwslt/NMTGMinor')
print(sys.path)
i = 0
while i < len(sys.path):
    if 'quan' in sys.path[i]:
        del sys.path[i]
        continue
    i += 1

print(sys.path)

from NMTGMinor.translate_api import add_parser_args, TranlateAPI

try:
    from simuleval import READ_ACTION, WRITE_ACTION, DEFAULT_EOS
    from simuleval.agents import SpeechAgent
    from simuleval.states import ListEntry, SpeechStates
except ImportError:
    print("Please install simuleval 'pip install simuleval'")


BOW_PREFIX = "\u2581"
BPE_SUFFIX = "@@"
BOS = 0
EOS = 2

def token2word(ids, dict):
    res = []
    for i in ids:
        t = dict[i]
        if t.endswith(BPE_SUFFIX):
            t = t.replace(BPE_SUFFIX, "")
        else:
            t += " "
        res.append(t)
    return res

class PynnAgent(SpeechAgent):

    def __init__(self, args):
        super().__init__(args)
        print(args)

        self.args = args
        if args.debug:
            print("Initialize the model...")
        self.model = TranlateAPI(args)
        self.dic = self.model.dict()
        if args.debug:
            print("Done.")

        self.chunk_len = args.chunk_len
        self.max_len = args.max_len
        self.debug = args.debug
        self.hold_n = args.hold_n
        self.la_n = args.la_n
        self.beam_agree = args.beam_agree
        self.bos_token = args.bos_token
        self.jazh = args.jazh
        self.max_context = args.max_context

    def build_states(self, args, client, sentence_id):
        states = SpeechStates(args, client, sentence_id, self)
        self.initialize_states(states)
        return states

    def initialize_states(self, states):
        states.units.target = ListEntry()
        states.src = []
        states.chunks_hyp = []
        states.displayed = []
        states.retrieved = []
        states.new_segment = False
        states.write = []
        self.bos = True

    @staticmethod
    def add_args(parser):
        add_parser_args(parser)
        
        parser.add_argument('--chunk-len', help='chunk len in ms', type=int, default=500)
        parser.add_argument('--max-len', help='max len', type=int, default=400)
        parser.add_argument('--debug', help='debug', action='store_true')

        parser.add_argument('--encoding', type=str, default='bpe')
        parser.add_argument('--hold-n', help='hold-n strategy', type=int, default=None)    
        parser.add_argument('--la-n', help='LA-n strategy', type=int, default=None)    
        parser.add_argument('--beam-agree', action='store_true')
        parser.add_argument('--jazh', action='store_true') 
        parser.add_argument('--max-context', type=int, default=20)  


    def segment_to_units(self, segment, states):
        states.retrieved += segment
        return [segment]

    def _cut_condition(self, suffix):
        punct = [any(p in self.dic[t] for p in list('.!?')) for t in suffix]
        try:
            i = punct.index(True)
            return i < len(suffix)/2
        except:
            return False

    def _index_to_cut(self, states):
        if len(states.displayed) > 0:
            dsuff = []
            for i in reversed(range(len(states.displayed))):
                dsuff += states.displayed[i]
                if self._cut_condition(dsuff):
                    return i
        return 0

    def update_states_read(self, states):
        if len(states.retrieved) / 16 >= self.chunk_len or states.finish_read():
            l = len(states.retrieved) if states.finish_read() else self.chunk_len * 16

            #cut = self._index_to_cut(states)
            assert len(states.src) == len(states.displayed)
            cut = max(0, len(states.src) - 10)
            print(f'CUT: {cut}')
            if cut > 0:
                states.src = states.src[cut:]
                states.displayed = states.displayed[cut:]
                # for i in range(len(states.displayed)):
                #     punct = [any(p in self.dic[t] for p in list('.!?')) for t in states.displayed[i]]
                #     if any(punct):
                #         states.displayed[i] = states.displayed[i][punct.index(True):]
                #         break

            states.src.append(states.retrieved[:l])
            states.retrieved = states.retrieved[l:]
            states.new_segment = True
    
    def prefix(self, states):
        if states.finish_read() and len(states.chunks_hyp) > 0:
            return states.chunks_hyp[-1]

        if len(states.chunks_hyp) < self.la_n:
            return []
        else:            
            undisplayed_suffixes = states.chunks_hyp[-self.la_n:]

            undisplayed_suffixes = [s for s in undisplayed_suffixes[-self.la_n:]]
            prefix = []
            for cs in zip(*undisplayed_suffixes):
                c = cs[0]
                if all(o == c for o in cs) and c != EOS:
                    prefix.append(c)
                else: 
                    break

            return prefix

    def remove_bos(self, tokens):
        if len(tokens) > 0 and tokens[0] == self.bos:
            return tokens[1:]
        else:
            return tokens

    def get_displayed(self, states):
        if len(states.displayed) > 0:
            return np.concatenate(states.displayed, axis=0)
        return []

    def _predict(self, states):
        states.new_segment = False

        #hypo, _, _, _, _ = decode(self.model, self.device, self.args, src, None, prefix=prefix)
        
        src = torch.FloatTensor(np.concatenate(states.src, axis=0))
        src /= (1 << 31)
        src /= src.abs().max()
        src = src.unsqueeze(1)

        displayed = self.get_displayed(states)
        displayed = self.remove_bos(displayed)
        prefix = [torch.LongTensor(displayed)] if len(displayed) > 0 else None
        _, hypo = self.model.infer(src, prefix, 'wav')
        
        hypo = hypo[0][0].cpu().numpy()
        hypo = self.remove_bos(hypo)
        hypo = hypo[len(prefix[0]):] if prefix else hypo
        print('prefix : ', prefix)
        print('hypo   : ', hypo)
        
        states.chunks_hyp.append(hypo)
        p = self.prefix(states)
        print('nprefix: ', p)
        print()
        states.displayed.append(p)

        if self.debug:
            if len(states.chunks_hyp) > 1:
                for i, h in list(enumerate(states.chunks_hyp))[-2:]:
                    hd = "".join(token2word(h, self.dic))
                    print(f'\t{i} - H - {h}')
                    print(f'\t{i} - H - {hd}')
        if len(p) > 0:
            for h in range(self.la_n):
                states.chunks_hyp[-h-1] = states.chunks_hyp[-h-1][len(p):]

            if self.debug:
                s = "".join(token2word(p, self.dic))
                print(f'{len(states.chunks_hyp)} - P - {p}')
                print(f'{len(states.chunks_hyp)} - P - {s}')

            states.write = p

            if self.debug:
                displayed = np.concatenate(states.displayed, axis=0) if len(states.displayed) > 1 else (states.displayed[0] if len(states.displayed) > 0 else [])
                d = "".join(token2word(displayed, self.dic))
                print(f'{len(states.chunks_hyp)} - D - {d}')
                print()
            return True
        return False

    @staticmethod
    def is_punct(t):
        return '.' in t or '?' in t or '!' in t

    def units_to_segment(self, units, states):
        # Merge sub word to full word.
        if EOS == units[0]:
            return DEFAULT_EOS

        segment = []
        if None in units.value:
            units.value.remove(None)

        for index in units:
            if index is None:
                units.pop()
            token = self.dic[index]
            

            if self.jazh:
                if index == EOS:
                    return [DEFAULT_EOS]
                units.pop()
                return [token.replace(BOW_PREFIX, " ")]

            if token.startswith(BOW_PREFIX) or index == EOS:
                if len(segment) == 0:
                    if token != EOS:
                        segment += [token.replace(BOW_PREFIX, "")]
                    else:
                        segment += [DEFAULT_EOS]
                else:
                    for j in range(len(segment)):
                        units.pop()

                    string_to_return = ["".join(segment)]
                    if self.bos:
                        string_to_return[0] = string_to_return[0][0].upper() + string_to_return[0][1:]
                        self.bos = False

                    if EOS == units[0]:
                        string_to_return += [DEFAULT_EOS]

                    self.bos = self.is_punct(string_to_return[-1])

                    return string_to_return
            else:
                segment += [token.replace(BOW_PREFIX, "")]

        if (
            len(units) > 0
            and EOS == units[-1]
            or len(states.units.target) > self.max_len
        ):
            tokens = [self.dic[unit] for unit in units if unit != EOS]
            return ["".join(tokens).replace(BOW_PREFIX, "")] + [DEFAULT_EOS]

        return None

    def policy(self, states):
        if len(states.write) > 0:
            return WRITE_ACTION
        if states.new_segment:
            if self._predict(states): return WRITE_ACTION
        if states.finish_read(): return WRITE_ACTION
        return READ_ACTION

    def predict(self, states):
        if len(states.write) == 0:
            return EOS
        w = states.write[0]
        states.write = states.write[1:]
        return w