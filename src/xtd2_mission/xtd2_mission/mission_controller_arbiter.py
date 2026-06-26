#!/usr/bin/env python3
"""Arbiter-enabled mission_controller entry point.

Existing launch files still run the console script named ``mission_controller``.
This wrapper keeps that name usable while enabling the single-output command
architecture by default:

1. set COMMAND_ARBITER_ENABLE=1 unless the user explicitly disables it;
2. start one command_arbiter process with the same num_drones parameter;
3. call the original mission_controller.main().
"""

import atexit
import os
import re
import subprocess
import sys

from xtd2_mission import mission_controller as mission_controller_core


def _find_params_file(argv):
    for idx, item in enumerate(argv):
        if item == '--params-file' and idx + 1 < len(argv):
            return argv[idx + 1]
        if item.startswith('--params-file='):
            return item.split('=', 1)[1]
    return None


def _read_num_drones_from_params(path, default=1):
    if not path or not os.path.exists(path):
        return int(os.environ.get('COMMAND_ARBITER_NUM_DRONES', default))
    try:
        text = open(path, 'r', encoding='utf-8').read()
    except Exception:
        return int(os.environ.get('COMMAND_ARBITER_NUM_DRONES', default))
    match = re.search(r'^\s*num_drones\s*:\s*(\d+)\s*$', text, re.MULTILINE)
    if match:
        return int(match.group(1))
    return int(os.environ.get('COMMAND_ARBITER_NUM_DRONES', default))


def _start_command_arbiter(num_drones):
    if os.environ.get('COMMAND_ARBITER_AUTOSTART', '1').strip().lower() not in ('1', 'true', 'yes', 'on'):
        return None
    cmd = [
        'ros2',
        'run',
        'xtd2_mission',
        'command_arbiter',
        '--num-drones',
        str(int(num_drones)),
        '--max-age-sec',
        os.environ.get('COMMAND_ARBITER_MAX_AGE_SEC', '0.8'),
        '--rate',
        os.environ.get('COMMAND_ARBITER_RATE', '20'),
        '--escape-hold-sec',
        os.environ.get('COMMAND_ARBITER_ESCAPE_HOLD_SEC', '0.8'),
        '--recovery-hold-sec',
        os.environ.get('COMMAND_ARBITER_RECOVERY_HOLD_SEC', '0.8'),
    ]
    proc = subprocess.Popen(cmd, env=os.environ.copy())

    def _cleanup():
        if proc.poll() is None:
            proc.terminate()
    atexit.register(_cleanup)
    return proc


def main():
    os.environ.setdefault('COMMAND_ARBITER_ENABLE', '1')
    num_drones = _read_num_drones_from_params(_find_params_file(sys.argv), default=1)
    _start_command_arbiter(num_drones)
    return mission_controller_core.main()


if __name__ == '__main__':
    main()
