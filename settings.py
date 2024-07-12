import os
wd = os.path.dirname(__file__)
figpath = os.path.join(wd, "figures")

if os.path.isdir(figpath)==False:
    os.system(f"mkdir {figpath}")