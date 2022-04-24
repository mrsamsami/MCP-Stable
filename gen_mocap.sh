#!/bin/bash

source ~/ENV/bin/activate

# python drloco/render_expert_ant.py --direction 0
# python drloco/render_expert_ant.py --direction 1
# python drloco/render_expert_ant.py --direction 2
# python drloco/render_expert_ant.py --direction 3


python test.py --direction 0
python test.py --direction 1
python test.py --direction 2
python test.py --direction 3