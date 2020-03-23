import nilmtk
from nilmtk import DataSet
from nilmtk.dataset_converters.refit.convert_refit import convert_refit
import os
from File_Management.path_and_file_management_Func import try_to_find_file

load_disaggregation_path_ = 'E:\OneDrive_Extra\Database\Load_Disaggregation'
ampds2_dataset = DataSet(os.path.join(load_disaggregation_path_,
                                      'AMPds2\dataverse_files\AMPds2.h5'))

# convert_refit(input_path=os.path.join(load_disaggregation_path_,
#                                       'REFIT\Cleaned\CLEAN_REFIT_081116'),
#               output_filename=os.path.join(load_disaggregation_path_,
#                                            'REFIT\Cleaned\CLEAN_REFIT_081116\Refit.h5'),
#               format='HDF')

refit_dataset = DataSet(os.path.join(load_disaggregation_path_,
                                     'REFIT\Cleaned\CLEAN_REFIT_081116\Refit.h5'))
