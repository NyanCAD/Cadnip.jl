import os

def run(settings):
    vafile = os.path.join(settings["source_dir"], "..", "..", "..", "..", "devices", "spice", "jfet1.va")
    c1 = settings["run_openvaf"](vafile, "jfet1.osdi")

    vafile = os.path.join(settings["source_dir"], "..", "..", "..", "..", "devices", "spice", "resistor.va")
    c2 = settings["run_openvaf"](vafile, "resistor.osdi")
    
    vafile = os.path.join(settings["source_dir"], "..", "..", "..", "..", "devices", "spice", "capacitor.va")
    c3 = settings["run_openvaf"](vafile, "capacitor.osdi")
    
    return c1 and c2 and c3
    