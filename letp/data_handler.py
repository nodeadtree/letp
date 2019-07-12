class DataHandler(object):

    """This is the interface for data"""

    def __init__(self, data,
                 reader=lambda x: x,
                 adder= lambda x,y: x if not x.append(y) else x,
                 partitioner=lambda x: x
                 ):
        """Instantiates a DataHandler object with some kind of data
            :data: this is used as a parameter for reader, which gives the actual data
            :adder: function responsible for collecting new data
            :partitioner: determines how data is to be manipulated
        """
        self.reader = reader
        self.adder = adder
        self.partitioner = partitioner
        self._data = self.reader(data)
        """
        if adder is None:
            def adder(big_data, little_data):
                big_data.append(little_data)
                return big_data
        """

    def add_data(self, data):
        """Adds data to internal data representation

        :data: some data to be added to the internal data
        :returns: returns count of self._data

        """
        self._data = self.adder(self._data, data)
        return len(self._data)

    def add_raw_data(self, data):
        """Adds data to internal data representation

        :data: raw data in need of being read by the reader
        :returns: returns count of self._data

        """
        self._data = self.adder(self._data, self.reader(data))
        return len(self._data)

    def __iter__(self):
        return iter(self.partitioner(self._data))
