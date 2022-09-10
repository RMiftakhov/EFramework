from abc import ABC, abstractmethod
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
        return self.vm



class SegyIO3D(SeismicData):

    def __init__(self, file_name, iline_byte=189, xline_byte=193):
        self._segyfile = segyio.open(file_name, iline=int(iline_byte), xline=int(xline_byte))

        # get statistics for visualization
        n_slices = 10
        seis = [self.get_iline(random.randint(0, self.get_n_ilines()-1)) for i in range(n_slices)]
        self.vm = np.percentile(seis, 95)

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

class SegyIO2D(SeismicData):

    def __init__(self, file_name):
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

    def __del__(self):
        pass

    def get_iline(self):
        return self._data.T


    def get_xline(self,):
        pass

    def get_zslice(self,):
        pass
    
    # get total number of ilines
    def get_n_ilines(self):
        return self._data.shape[0]

    # get total number of xlines
    def get_n_xlines(self):
        pass

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

class Numpy2D(SeismicData):

    def __init__(self, file_name):
        self._data = np.load(file_name)

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
        return self._data.shape[0]

    # get total number of xlines
    def get_n_xlines(self):
        pass

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