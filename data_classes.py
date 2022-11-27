from abc import ABC, abstractmethod
from unittest import result
import segyio
import numpy as np
import random
random.seed(1)

class SeismicData(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __del__(self):
        pass

    @abstractmethod
    def get_iline(self, indx):
        pass

    @abstractmethod
    def get_xline(self, indx):
        pass

    @abstractmethod
    def get_zslice(self, indx):
        pass
    
    # get total number of ilines
    @abstractmethod
    def get_n_ilines(self):
        pass

    # get total number of xlines
    @abstractmethod
    def get_n_xlines(self):
        pass

    # get total number of zslices
    @abstractmethod
    def get_n_zslices(self):
        pass

    @abstractmethod
    def get_sample_rate(self):
        pass

    @abstractmethod
    def get_vm(self):
        pass

class SegyIO3D(SeismicData):

    def __init__(self, file_name, iline_byte=189, xline_byte=193):
        super().__init__()
        self._segyfile = segyio.open(file_name, iline=int(iline_byte), xline=int(xline_byte))

        # get statistics for visualization
        n_slices = 10
        seis = [self.get_iline(random.randint(0, self.get_n_ilines()-1)) for i in range(n_slices)]
        self.vm = np.percentile(seis, 95)
        self.file_name = file_name
        self.iline_byte = iline_byte
        self.xline_byte = xline_byte

    def __del__(self):
        self._segyfile.close()

    def get_iline(self, indx):
        return self._segyfile.iline[self._segyfile.ilines[indx]]


    def get_xline(self,indx):
        return self._segyfile.xline[self._segyfile.xlines[indx]]


    def get_zslice(self,indx):
        return self._segyfile.depth_slice[indx]
    
    # get total number of ilines
    def get_n_ilines(self):
        return len(self._segyfile.iline)

    # get total number of xlines
    def get_n_xlines(self):
        return len(self._segyfile.xline)

    # get total number of zslices
    def get_n_zslices(self):
        return self._segyfile.samples.size

    def get_sample_rate(self):
        return segyio.tools.dt(self._segyfile)

    def get_vm(self):
        return self.vm

    def get_file_name(self):
        return self.file_name

    def cropped_numpy(self, min_il, max_il, min_xl, max_xl, min_z, max_z):
        """ Reads cropped seismic and returns numpy array

        Args:
            min_il (_type_): min inline
            max_il (_type_): max inline
            min_xl (_type_): min crossline
            max_xl (_type_): max crossline
            min_z (_type_): min timeslice
            max_z (_type_): max timeslice
        """
        assert max_il>min_il, f"max_il must be greater than {min_il}, got: {max_il}"
        assert max_xl>min_xl, f"max_il must be greater than {min_xl}, got: {max_xl}"
        assert max_z>min_z,   f"max_il must be greater than {min_z}, got: {max_z}"

        return np.array([self.get_iline(i)[min_xl:max_xl, min_z:max_z] for i in range(min_il, max_il)])

    def get_xline_byte(self):
        return self.xline_byte

    def get_iline_byte(self):
        return self.iline_byte

    def get_str_format(self):
        return "SEGY"

    def get_str_dim(self):
        return "3D"


class SegyIO2D(SeismicData):

    def __init__(self, file_name):
        super().__init__()
        seismic_type = "2D"
        try: 
            with segyio.open(file_name, strict=True) as segyfile:
                seismic_type = "3D"
                raise ValueError("You can only use 2D seismic file with this mode")
        except:
            if seismic_type == "3D":
                raise ValueError("You can only use 2D seismic file with this mode")
            with segyio.open(file_name, strict=False) as segyfile:
                self._data = np.stack(list((_.copy() for _ in segyfile.trace[:])))
                self._dt = segyio.tools.dt(segyfile)
                self.vm = np.percentile(self._data, 95)
                self.file_name = file_name


    def __del__(self):
        pass

    def get_file_name(self):
        return self.file_name

    def get_iline(self):
        return self._data.T


    def get_xline(self,):
        pass

    def get_zslice(self,):
        pass
    
    # get total number of ilines
    def get_n_ilines(self):
        pass

    # get total number of xlines
    def get_n_xlines(self):
        return self._data.shape[0]

    # get total number of zslices
    def get_n_zslices(self):
        return self._data.shape[1]

    def get_sample_rate(self):
        return self._dt

    def get_vm(self):
        return self.vm

    def make_axis_devisable_by(self, factor):
        xlim = self._data.shape[0]//int(factor)*int(factor)
        ylim = self._data.shape[1]//int(factor)*int(factor)
        self._data = self._data[:xlim, :ylim]
    
    def get_xline_byte(self):
        return self.xline_byte

    def get_iline_byte(self):
        return self.iline_byte

    def get_str_format(self):
        return "SEGY"

    def get_str_dim(self):
        return "2D"


class Numpy2D(SeismicData):

    def __init__(self, data):
        super().__init__()
        if isinstance(data, str):
            self._data = np.load(data)
        elif isinstance(data, np.ndarray):
            self._data = data

        # get statistics for visualization
        seis = self.get_iline()
        self.vm = np.percentile(seis, 95)

    def __del__(self):
        pass

    def get_iline(self):
        return self._data

    def get_xline(self):
        pass

    def get_zslice(self):
        pass

    def get_n_ilines(self):
        pass

    # get total number of xlines
    def get_n_xlines(self):
        return self._data.shape[0]

    def get_n_zslices(self):
        return self._data.shape[1]

    def get_sample_rate(self):
        return 1000

    def get_vm(self):
        return self.vm

    def make_axis_devisable_by(self, factor):
        xlim = self._data.shape[0]//int(factor)*int(factor)
        ylim = self._data.shape[1]//int(factor)*int(factor)
        self._data = self._data[:xlim, :ylim]

    def get_str_format(self):
        return "NUMPY"

    def get_str_dim(self):
        return "2D"

class Numpy3D(SeismicData):

    def __init__(self, data):
        super().__init__()
        if isinstance(data, str):
            self._data = np.load(data)
        elif isinstance(data, np.ndarray):
            self._data = data

        # get statistics for visualization
        if self._data is not None:
            n_slices = 10
            seis = [self.get_iline(random.randint(0, self.get_n_ilines()-1)) for i in range(n_slices)]
            self.vm = np.percentile(seis, 95)

    def __del__(self):
        pass

    def get_iline(self, indx):
        return self._data[indx,:,:]

    def get_xline(self, indx):
        return self._data[:,indx,:]

    def get_zslice(self, indx):
        return self._data[:,:,indx]

    def get_n_ilines(self):
        return self._data.shape[0]

    # get total number of xlines
    def get_n_xlines(self):
        return self._data.shape[1]

    def get_n_zslices(self):
        return self._data.shape[2]

    def get_sample_rate(self):
        return 1000

    def get_vm(self):
        return self.vm

    def get_cube(self):
        return self._data

    def make_axis_devisable_by(self, factor):
        xlim = self._data.shape[0]//int(factor)*int(factor)
        ylim = self._data.shape[1]//int(factor)*int(factor)
        self._data = self._data[:xlim, :ylim, :]

    def get_str_format(self):
        return "NUMPY"
    
    def get_str_dim(self):
        return "3D"
