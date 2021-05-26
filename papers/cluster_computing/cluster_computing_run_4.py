source_code = """
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

cwd = Path().cwd().parent
spec = spec_from_file_location("Energies_2020", cwd / 'Energies_2020/stage_fft.py')
foo = module_from_spec(spec)
spec.loader.exec_module(foo)
foo.get_freq_corr_hst_and_sum_of_sel_freq("AMPDS2", "temperature")

"""
if __name__ == '__main__':
    exec(source_code)
