from copy import deepcopy
import pytest

from loghound_agent_debug.adapters.codetracebench_context import miniswe_context,openhands_context,terminus_context,native_predictions


def test_miniswe_actions_not_all_messages_and_no_annotation_fields():
    raw={'messages':[{'role':'user','content':'TASK'}, {'role':'assistant','content':'just thinking'},
         {'role':'assistant','content':'THOUGHT\n```bash\necho hello\n```'},
         {'role':'user','content':'<returncode>0</returncode>hello'}], 'incorrect_stages':'SECRET','info':{'answer':'SECRET'}}
    p=miniswe_context(raw)
    assert len(p['case']['steps'])==1 and p['native_step_ids']==['1']
    assert p['case']['steps'][0]['action']=='echo hello'
    assert 'SECRET' not in str(p)
    other=deepcopy(raw);other['incorrect_stages']='OTHER';assert miniswe_context(other)==p


def test_openhands_pairs_observation_by_cause_and_uses_native_action_sequence():
    events=[{'id':0,'action':'message','message':'task'},
      {'id':5,'action':'run','message':'Running command: echo a','args':{'thought':'plan'}},
      {'id':9,'action':'run_ipython','args':{'code':'print(2)'}},
      {'id':12,'cause':9,'observation':'run_ipython','content':'2'},
      {'id':7,'cause':5,'observation':'run','content':'a'}]
    p=openhands_context(events,'task')
    assert [s['observation'] for s in p['case']['steps']]==['a','2']
    assert p['native_step_ids']==['1','2']
    assert p['provenance'][1]['action_event_id']==9


def test_terminus_uses_next_prompt_and_keeps_gapped_native_ids():
    episodes={0:{'response':'{"commands":[{"keystrokes":"ls"}],"analysis":"inspect"}'},
              1:{'prompt':'New Terminal Output:\nfile'},3:{'response':'finished'}}
    p=terminus_context(episodes,'task')
    assert p['native_step_ids']==['1','4']
    assert p['case']['steps'][0]['observation']=='file'
    assert native_predictions(['1','0'],p)==['4','1']
    with pytest.raises(ValueError):native_predictions(['2'],p)


def test_openhands_rejects_unmapped_file_actions_instead_of_shifting_ids():
    with pytest.raises(ValueError, match='Unverified OpenHands native action mapping'):
        openhands_context([{'id':3,'action':'read','args':{'path':'file.py'}},
                           {'id':5,'action':'run','message':'Running command: cat file.py'}], 'task')
