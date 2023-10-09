import os

OUTPUT_BASE = "output"


def output_directory(name):
    directory = os.path.join(OUTPUT_BASE, name)
    os.makedirs(directory, exist_ok=True)

    def complete_filename(base_name):
        return os.path.join(directory, base_name)

    return complete_filename
