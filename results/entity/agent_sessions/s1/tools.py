"""Agent tools for this session. Python: `import sys; sys.path.insert(0, "/home/ubuntu/wp/runs/entity/agent/s1"); import tools as T`, then help(T.generate) etc.
Command line: `python tools.py --help`. Implementation: /home/ubuntu/wp/patchkit/agent_tools.py (read-only)."""
import os
import sys

os.environ['AGENT_SESSION'] = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"/home/ubuntu/wp")
from patchkit.agent_tools import *  # noqa: F401,F403,E402
from patchkit.agent_tools import __all__, main  # noqa: F401,E402

if __name__ == '__main__':
    main()
