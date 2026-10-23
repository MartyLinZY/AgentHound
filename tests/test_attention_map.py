from dataclasses import replace
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from loghound_agent_debug.rankers.attention_map import AttentionMapRanker, listwise_loss
from loghound_agent_debug.features.attention_inputs import build_inputs, grouped_folds, text_vector
from loghound_agent_debug.schema import Step, TrajectoryInput


def item(repo='repo', task='task', tid='t', n=4):
    return TrajectoryInput('1', 'rootse', tid, 'fix parser', 'coding', 'failure', [],
        [Step(i, str(i), [str(i)], 'a', 'assistant', '', decision_text=f'parse value {i}') for i in range(n)],
        repository=repo, task_id=task)


def tensors(n=4):
    torch.manual_seed(1)
    x=torch.randn(n, 521);f=torch.randn(256);e=torch.zeros(n,n)
    if n>1:e[1,0]=1
    return x,f,e


def test_time_direction_and_missing_edge_support():
    x,f,e=tensors();dense=AttentionMapRanker(521);fixed=AttentionMapRanker(521,variant='fixed_graph')
    fixed.load_state_dict(dense.state_dict())
    a=dense(x,f,e)['attention'];b=fixed(x,f,e)['attention']
    assert torch.allclose(a.sum(1),torch.ones(4))
    assert not torch.triu(a,diagonal=1).any()
    assert a[3,0]>0 and b[3,0]==0
    assert torch.isfinite(dense(x,f,torch.zeros_like(e))['scores']).all()


def test_failure_condition_changes_map_and_unconditioned_map_ignores_it():
    x,f,e=tensors();model=AttentionMapRanker(521)
    assert not torch.allclose(model(x,f,e)['attention'],model(x,-f,e)['attention'])
    model.variant='dense_unconditioned'
    assert torch.equal(model(x,f,e)['attention'],model(x,-f,e)['attention'])


def test_local_residual_survives_and_map_affects_scores():
    x,f,e=tensors();model=AttentionMapRanker(521)
    out=model(x,f,e);shuf=model(x,f,e,shuffle_seed=7)
    assert torch.allclose(out['scores'],out['local']+out['relation'])
    assert not torch.allclose(out['scores'],shuf['scores'])
    assert torch.allclose(shuf['attention'].sum(1),torch.ones(4))
    assert torch.allclose(torch.sort(shuf['attention'],dim=1).values,torch.sort(out['attention'],dim=1).values)


def test_no_prior_variant_ignores_graph():
    x,f,e=tensors();model=AttentionMapRanker(521,variant='dense_no_prior')
    assert torch.equal(model(x,f,e)['scores'],model(x,f,torch.zeros_like(e))['scores'])


def test_single_step_and_training_gradients():
    x,f,e=tensors(1);m=AttentionMapRanker(521)
    assert torch.isfinite(m(x,f,e)['scores']).all()
    x,f,e=tensors();loss=listwise_loss(m(x,f,e)['scores'],[0]);loss.backward()
    assert m.query.weight.grad.abs().sum()>0
    assert m.failure_query.weight.grad.abs().sum()>0
    with pytest.raises(ValueError):listwise_loss(torch.ones(4),[4])
    with pytest.raises(ValueError):m(x,f,e.T)


def test_feature_encoder_is_deterministic_and_preserves_full_text():
    a=item();numeric=np.zeros((4,5));x=build_inputs(a,[(0,2)],numeric)
    y=build_inputs(a,[(0,2)],numeric)
    assert np.array_equal(x.x,y.x) and x.prior[2,0]==1 and x.prior[0,2]==0
    assert not np.array_equal(text_vector('x '*10000),text_vector('x '*10000+' unique_suffix'))
    with pytest.raises(ValueError):build_inputs(a,[(2,0)],numeric)
    with pytest.raises(ValueError):build_inputs(a,[],np.zeros((3,5)))


def test_repository_and_task_groups_cannot_cross_folds():
    items=[item('a','t1','1'),item('a','t2','2'),item('b','t2','3'),item('c','t3','4'),item('d','t4','5')]
    folds,groups=grouped_folds(items,3)
    assert folds[0]==folds[1]==folds[2] and groups[0]==groups[2]
    assert len(set(folds))==3
    assert grouped_folds(items,3)==(folds,groups)
    with pytest.raises(ValueError):grouped_folds([replace(items[0],repository='')],2)
