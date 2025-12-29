import os

def run(settings):
    vafile = os.path.join(settings["source_dir"], "..", "..", "..", "..", "devices", "spice", "jfet1.va")
    return settings["run_openvaf"](vafile, "jfet1.osdi")
    