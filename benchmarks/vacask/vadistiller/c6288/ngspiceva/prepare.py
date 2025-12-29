import os

def run(settings):
    vafile = os.path.join(settings["source_dir"], "..", "..", "..", "..", "devices", "spice", "bsim3v3.va")
    return settings["run_openvaf"](vafile, "bsim3v3.osdi")
    
