import os, shutil

def run(settings):
    vafile = os.path.join(settings["source_dir"], "..", "..", "..", "..", "devices", "bsim3v3.va")
    shutil.copy(vafile, ".")
    with open("bsim3v3.va", "r") as f:
        txt = f.read()
    txt = txt.replace("module bsim3(d, g, s, b);", "module bsim3cg(d, g, s, b);")
    with open("bsim3v3.va", "w") as f:
        f.write(txt)
    return settings["run_openvaf"]("bsim3v3.va", "bsim3v3.osdi")
    
