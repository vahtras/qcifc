from abc import ABCMeta


class QuantumChemistry(object):

    @staticmethod
    def get_factory(code, workdir):
        if code == 'Dalton':
            return DaltonFactory(workdir)
        else:
            raise TypeError, "QM %s not implemented" % code


class DaltonFactory(QuantumChemistry):

    def __init__(self, tmpdir):
        self.__tmpdir = tmpdir

    def get_workdir(self):
        return self.__tmpdir
