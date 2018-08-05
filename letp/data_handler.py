class DataHandler(object):

    """This is the interface for data"""

    def __init__(self, data, partitioner=lambda x: x, adder=None):
        """Instantiates a DataHandler object with some kind of data

        :data: TODO

        """
        self.partitioner = partitioner
        if adder is None:
            def adder(big_data, little_data):
                big_data.append(little_data)
                return big_data
        self.adder = adder
        self._data = data

    def add_data(self, data):
        """Adds data to internal data representation

        :data: TODO
        :returns: TODO

        """
        self._data = self.adder(self._data, data)

    def __iter__(self):
        return iter(self.partitioner(self._data))
