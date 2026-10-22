import json
import numpy as np
from bsl.sentence import SentenceRecognizer

def test_background_target_never_becomes_a_caption(tmp_path):
    config={'backend':'dtw','threshold':.1,'labels':['হ্যালো','ধন্যবাদ'],'feature_variant':'shape-motion'}
    (tmp_path/'sentences.json').write_text(json.dumps(config),encoding='utf-8')
    np.savez(tmp_path/'sentences-dtw.npz',templates=np.zeros((1,32,104),np.float32),targets=np.array([2]))
    result=SentenceRecognizer(tmp_path)(np.zeros((20,98),np.float32))
    assert result['label']=='Unrecognised' and not result['accepted']

def test_temporal_neural_model_loads_with_matching_input_contract(tmp_path):
    from tensorflow.keras import Sequential,Input
    from tensorflow.keras.layers import GlobalAveragePooling1D,Dense
    model=Sequential([Input((32,104)),GlobalAveragePooling1D(),Dense(3,activation='softmax')])
    model.layers[-1].set_weights([np.zeros((104,3),np.float32),np.array([10,0,0],np.float32)])
    model.save(tmp_path/'sentences.keras')
    config={'backend':'gru','threshold':.8,'labels':['হ্যালো','ধন্যবাদ'],'feature_variant':'shape-motion'}
    (tmp_path/'sentences.json').write_text(json.dumps(config),encoding='utf-8')
    result=SentenceRecognizer(tmp_path)(np.zeros((20,98),np.float32))
    assert result['accepted'] and result['label']=='হ্যালো'
