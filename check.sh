python3 -c "
try:
    from vllm._rs import compute_running_tokens
    print('OK: vllm._rs')
except ImportError:
    try:
        from _rs import compute_running_tokens
        print('OK: _rs (dev mode)')
    except ImportError:
        print('FAIL: not installed')
"
