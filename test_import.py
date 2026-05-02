import sys
import io

# Capture stdout/stderr
old_stdout = sys.stdout
old_stderr = sys.stderr
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

try:
    import test_modem_simple
    success = True
    output = sys.stdout.getvalue()
    errors = sys.stderr.getvalue()
except Exception as e:
    success = False
    output = str(e)
finally:
    sys.stdout = old_stdout
    sys.stderr = old_stderr

if success:
    print('Module imported successfully')
    print('preamble_td exists:', hasattr(test_modem_simple, 'preamble_td') and test_modem_simple.preamble_td is not None)
    print('build_tx_audio_from_bytes exists:', hasattr(test_modem_simple, 'build_tx_audio_from_bytes'))
    print('receive_from_wav exists:', hasattr(test_modem_simple, 'receive_from_wav'))
    print('live_receive_and_process exists:', hasattr(test_modem_simple, 'live_receive_and_process'))
else:
    print('Import failed:', output)
