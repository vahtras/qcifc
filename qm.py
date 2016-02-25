import abc
import os

class QuantumChemistry(object):
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def get_factory(code, workdir):
        if code == 'Dalton':
            return DaltonFactory(workdir)
        else:
            raise TypeError, "QM %s not implemented" % code

    @abc.abstractmethod
    def get_overlap(self): pass

    @abc.abstractmethod
    def get_one_el_hamiltonian(self): pass
        


class DaltonFactory(QuantumChemistry):

    def __init__(self, tmpdir):
        self.__tmpdir = tmpdir

    def get_workdir(self):
        return self.__tmpdir

    def get_overlap(self):
        from daltools import one
        S = one.read(
            "OVERLAP", 
            os.path.join(self.get_workdir(), "AOONEINT")
            ).unpack().unblock()
        return S

    def get_one_el_hamiltonian(self):
        from daltools import one
        S = one.read(
            "ONEHAMIL", 
            os.path.join(self.get_workdir(), "AOONEINT")
            ).unpack().unblock()
        return S
