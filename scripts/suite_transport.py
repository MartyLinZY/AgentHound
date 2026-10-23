"""Audited, per-call resumable MoMA transport for frozen upstream algorithms."""
from __future__ import annotations
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import time
from datetime import datetime, timezone
import requests


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


class ExperimentAbort(BaseException):
    """Escape upstream catch-all handlers after a recorded terminal API failure."""
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


class RecordedClient:
    def __init__(self, out: Path, config: dict, key: str, post=requests.post, sleep=time.sleep):
        self.out, self.config, self.key = out, config, key
        self.post, self.sleep = post, sleep
        self.index = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def _read_stream(self, response, raw_path: Path) -> dict:
        """Persist SSE before parsing; keep reasoning separate from the final answer."""
        content, reasoning = [], []
        model, finish, usage = None, None, None
        done = False
        try:
            with raw_path.open('w') as raw:
                for line in response.iter_lines():
                    line = (line.decode('utf-8') if isinstance(line, bytes) else line).replace(self.key, '[REDACTED]')
                    raw.write(line + '\n'); raw.flush()
                    if not line.startswith('data:'):
                        continue
                    value = line[5:].strip()
                    if value == '[DONE]':
                        done = True
                        break
                    chunk = json.loads(value)
                    if chunk.get('model'):
                        if model and model != chunk['model']:
                            raise ValueError('Model changed during stream')
                        model = chunk['model']
                    if chunk.get('usage'):
                        usage = chunk['usage']
                    for choice in chunk.get('choices', []):
                        if choice.get('index', 0) != 0:
                            continue
                        delta = choice.get('delta', {})
                        if delta.get('content'):
                            content.append(delta['content'])
                        if delta.get('reasoning_content'):
                            reasoning.append(delta['reasoning_content'])
                        if choice.get('finish_reason'):
                            finish = choice['finish_reason']
            if not done or finish is None:
                raise ValueError('Incomplete SSE response')
            return {'model': model, 'choices': [{'index': 0, 'finish_reason': finish,
                    'message': {'role': 'assistant', 'content': ''.join(content) if content else None,
                                'reasoning_content': ''.join(reasoning)}}], 'usage': usage}
        finally:
            response.close()

    def _response(self, data: dict) -> SimpleNamespace:
        if data.get('model') != self.config['model']:
            raise ExperimentAbort('model_mismatch', 'Unexpected response model: ' + str(data.get('model')))
        choice = data['choices'][0]
        if choice.get('finish_reason') != 'stop':
            raise ExperimentAbort('output_truncated', 'Unfinished generation: ' + str(choice.get('finish_reason')))
        if not isinstance(choice['message'].get('content'), str):
            raise ExperimentAbort('empty_response', 'No final response content')
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=choice['message']['content']))],
                               usage=SimpleNamespace(**(data.get('usage') or {'total_tokens': 0})))

    def create(self, **kwargs) -> SimpleNamespace:
        if kwargs.get('model') != self.config['model'] or 'claude' in str(kwargs.get('model')).lower():
            raise ExperimentAbort('model_mismatch', 'Unapproved requested model')
        self.index += 1
        if self.index > self.config['max_calls_per_case']:
            raise ExperimentAbort('call_limit', 'Exceeded frozen safety bound')
        payload = {'model': self.config['model'], 'messages': kwargs['messages'],
                   'temperature': self.config['temperature'], 'max_tokens': self.config['max_tokens']}
        if self.config.get('stream'):
            payload.update(stream=True, stream_options={'include_usage': True},
                           enable_thinking=self.config.get('enable_thinking', True))
        call_dir = self.out / 'calls' / f'{self.index:04d}'
        state_path = call_dir / 'state.json'
        request_sha = digest(payload)
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state['request_sha256'] != request_sha:
                raise ExperimentAbort('replay_mismatch', 'Upstream request changed during replay')
        else:
            state = {'call_index': self.index, 'request_sha256': request_sha, 'status': 'prepared', 'attempts': [],
                     'upstream_caller': [frame.function for frame in inspect.stack()[1:5]],
                     'upstream_parameters': {k: v for k,v in kwargs.items() if k != 'messages'}}
            save(call_dir / 'request.json', payload)
            save(state_path, state)
        if digest(json.loads((call_dir / 'request.json').read_text())) != request_sha:
            raise ExperimentAbort('replay_mismatch', 'Saved request was modified')
        if state['status'] == 'received':
            return self._response(json.loads((call_dir / state['response_file']).read_text()))
        if state['status'] in {'context_overflow','service_error','transport_error','response_error'}:
            raise ExperimentAbort(state['status'], 'Replaying terminal failure')
        for attempt in state['attempts']:
            if attempt['status'] == 'started':
                response_file = call_dir / f"response-{attempt['number']}.json"
                if response_file.exists():
                    data = json.loads(response_file.read_text())
                    attempt.update(status='received', usage=data.get('usage'), recovered=True)
                    state.update(status='received', response_file=response_file.name)
                    save(state_path, state)
                    return self._response(data)
                attempt.update(status='interrupted_unknown', usage_unknown=True)
        while len(state['attempts']) < self.config['transport_attempts']:
            attempt = {'number': len(state['attempts'])+1, 'status': 'started', 'started_at': now()}
            state['attempts'].append(attempt)
            state['status'] = 'running'
            save(state_path, state)
            start = time.monotonic()
            retry = False
            try:
                response = self.post(self.config['endpoint'], headers={'Authorization': 'Bearer '+self.key},
                                     json=payload, timeout=(20,self.config['timeout_seconds']), allow_redirects=False,
                                     **({'stream': True} if self.config.get('stream') else {}))
                attempt.update(http_status=response.status_code, elapsed_seconds=time.monotonic()-start, finished_at=now())
                if response.status_code == 200:
                    if self.config.get('stream'):
                        data = self._read_stream(response, call_dir / f"response-{attempt['number']}.sse")
                    else:
                        body = response.text.replace(self.key, '[REDACTED]')
                        (call_dir / f"response-{attempt['number']}.txt").write_text(body)
                        data = json.loads(body)
                    attempt.update(elapsed_seconds=time.monotonic()-start, finished_at=now())
                    attempt.update(status='received', usage=data.get('usage'), response_model=data.get('model'))
                    filename = f"response-{attempt['number']}.json"
                    save(call_dir / filename, data)
                    state.update(status='received', response_file=filename)
                    save(state_path, state)
                    return self._response(data)
                error = response.text.replace(self.key,'[REDACTED]')[:3000]
                attempt.update(status='http_error', error=error, usage_unknown=True)
                state['status'] = 'context_overflow' if response.status_code==400 and any(x in error.lower() for x in ['context','input_tokens']) else 'service_error'
                retry = response.status_code==429 or response.status_code>=500
            except requests.RequestException as exc:
                attempt.update(status='transport_error', error_type=type(exc).__name__, usage_unknown=True,
                               elapsed_seconds=time.monotonic()-start, finished_at=now())
                state['status'] = 'transport_error'
                retry = True
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                attempt.update(status='response_error', error_type=type(exc).__name__,usage_unknown=True,
                               elapsed_seconds=time.monotonic()-start, finished_at=now())
                state['status'] = 'response_error'
            if retry and len(state['attempts']) < self.config['transport_attempts']:
                state['status'] = 'retry_pending'
            save(state_path,state)
            if state['status'] != 'retry_pending':
                raise ExperimentAbort(state['status'], 'Terminal API failure; see call state')
            self.sleep(5 * len(state['attempts']))
        state['status'] = 'transport_error'
        save(state_path,state)
        raise ExperimentAbort('transport_error','Transport attempt limit reached')
