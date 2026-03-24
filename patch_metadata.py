import os
import glob
import sys

# Locate the realtime dist-info metadata
site_packages_dirs = [p for p in sys.path if "site-packages" in p.lower()]
if not site_packages_dirs:
    print("Could not find site-packages directory.")
    sys.exit(1)

patched = False
for sp in site_packages_dirs:
    metadata_paths = glob.glob(os.path.join(sp, "realtime-*.dist-info", "METADATA"))
    for path in metadata_paths:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        if "websockets (<12.0" in content:
            new_content = content.replace(
                "websockets (<12.0,>=11.0.3)", "websockets (>=11.0.3)"
            )
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"Patched realtime METADATA at {path}")
            patched = True

if not patched:
    print("Could not find or didn't need to patch realtime METADATA.")
