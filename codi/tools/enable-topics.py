import os

topics = os.popen("gz topic --list").read().strip().split("\n")

for i in topics:
    if "enable_streaming" in i: 
        os.popen(f'gz topic -t {i} -m gz.msgs.Boolean -p "data:1"').read()
