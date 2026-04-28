#!/usr/bin/env python3
"""
MCP Server for SearXNG - local search engine integration
"""
import json
import urllib.request
import urllib.parse
import sys
from typing import Any

STDIN_BUFFER = sys.stdin.buffer
STDOUT_BUFFER = sys.stdout.buffer


def search(query: str, limit: int = 10) -> dict:
    """Search using local SearXNG instance"""
    try:
        encoded_query = urllib.parse.quote(query)
        url = f"http://127.0.0.1:8080/search?q={encoded_query}&format=json"

        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'SearXNG MCP Client')
        req.add_header('X-Forwarded-For', '127.0.0.1')

        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())

        results = []
        for item in data.get('results', [])[:limit]:
            results.append({
                'title': item.get('title', ''),
                'url': item.get('url', ''),
                'content': item.get('content', ''),
                'engine': item.get('engine', 'unknown'),
            })

        return {
            'success': True,
            'query': query,
            'count': len(results),
            'results': results
        }
    except urllib.error.URLError:
        return {
            'success': False,
            'error': 'SearXNG is not running at http://127.0.0.1:8080',
            'query': query
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'query': query
        }


def read_message() -> Any:
    headers = {}
    while True:
        line = STDIN_BUFFER.readline()
        if not line:
            return None
        line = line.decode('utf-8', 'replace').rstrip('\r\n')
        if line == '':
            break
        if ':' in line:
            key, value = line.split(':', 1)
            headers[key.strip().lower()] = value.strip()
        else:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    content_length = int(headers.get('content-length', '0'))
    if content_length <= 0:
        return None
    body = STDIN_BUFFER.read(content_length)
    if not body:
        return None
    return json.loads(body.decode('utf-8', 'replace'))


def write_message(message: dict) -> None:
    body = json.dumps(message, ensure_ascii=False).encode('utf-8')
    header = f'Content-Length: {len(body)}\r\n\r\n'.encode('utf-8')
    STDOUT_BUFFER.write(header)
    STDOUT_BUFFER.write(body)
    STDOUT_BUFFER.flush()


def build_response(request: dict, result: Any = None, error: dict | None = None) -> dict:
    response = {
        'jsonrpc': '2.0',
        'id': request.get('id', None)
    }
    if error is not None:
        response['error'] = error
    else:
        response['result'] = result
    return response


def handle_request(request: dict) -> Any:
    method = request.get('method')
    params = request.get('params', {})

    if method == 'initialize':
        return {
            'capabilities': {
                'search': True,
            }
        }

    if method == 'shutdown':
        return None

    if method in ('search', 'searxng.search', 'query'):
        if isinstance(params, str):
            query = params
        elif isinstance(params, dict):
            query = params.get('query') or params.get('q') or params.get('text') or ''
        else:
            query = ''

        if not query:
            raise ValueError('search requires a query string')

        limit = 10
        if isinstance(params, dict):
            limit = int(params.get('limit', limit))

        return search(query, limit)

    if method == 'ping':
        return 'pong'

    raise ValueError(f'Method not found: {method}')


def serve_stdin() -> None:
    running = True
    while running:
        request = read_message()
        if request is None:
            break

        response = None
        try:
            if isinstance(request, list):
                response = [build_response(req, result=handle_request(req) if 'id' in req else None)
                            for req in request]
            else:
                result = handle_request(request)
                response = build_response(request, result=result)
                if request.get('method') == 'shutdown':
                    running = False
        except Exception as exc:
            response = build_response(request, error={
                'code': -32000,
                'message': str(exc)
            })

        if response is not None:
            write_message(response)


def main() -> None:
    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])
        result = search(query)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        serve_stdin()


if __name__ == '__main__':
    main()
