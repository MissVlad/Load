source_code = """
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

cwd = Path().cwd()
spec = spec_from_file_location("Energies_Research_paper_2020", cwd / 'Energies_Research_paper_2020.py')
foo = module_from_spec(spec)
spec.loader.exec_module(foo)
foo.lasso_fft_and_correlation(task='explore', data_set=foo.TURKEY_HOUSE_DATA)

"""
if __name__ == '__main__':
    exec(source_code)
