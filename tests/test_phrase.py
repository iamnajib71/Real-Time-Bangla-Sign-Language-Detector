import numpy as np
from bsl.phrase import resample,dtw,PhraseRecognizer

def test_temporal_warp_and_unknown_rejection(tmp_path):
    a=np.linspace(0,1,32)[:,None]*np.ones((1,98))
    b=-a
    path=tmp_path/'phrases.npz'
    np.savez(path,templates=np.stack([a,b]),labels=np.array(['হ্যালো','ধন্যবাদ']))
    recognizer=PhraseRecognizer(path,threshold=.2,min_margin=.1)
    stretched=np.linspace(0,1,50)[:,None]*np.ones((1,98))
    result=recognizer(stretched)
    assert result['accepted'] and result['label']=='হ্যালো'
    assert recognizer(np.ones((32,98))*10)['label']=='Unrecognised'
    assert dtw(a,a)==0

def test_resampling_preserves_endpoints():
    a=np.array([[0,1],[2,3],[4,5]],np.float32)
    out=resample(a)
    assert out.shape==(32,2)
    np.testing.assert_array_equal(out[0],a[0]); np.testing.assert_array_equal(out[-1],a[-1])

def test_single_personal_phrase_can_be_recognised(tmp_path):
    a=np.linspace(0,1,32)[:,None]*np.ones((1,98))
    path=tmp_path/'personal.npz'
    np.savez(path,templates=a[None],labels=np.array(['হ্যালো']))
    result=PhraseRecognizer(path)(a)
    assert result['accepted'] and result['label']=='হ্যালো'
