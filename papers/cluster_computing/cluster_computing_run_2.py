source_code = """
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

cwd = Path().cwd().parent
spec = spec_from_file_location("Energies_2020", cwd / 'Energies_2020/stage_fft.py')
foo = module_from_spec(spec)
spec.loader.exec_module(foo)
foo.make_report("JOHN", 'solar')

"""
if __name__ == '__main__':
    exec(source_code)
