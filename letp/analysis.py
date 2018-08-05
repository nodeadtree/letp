import datetime
from json_tricks import dump
from os.path import join
from pathlib import Path


class Analyzer(object):

    """The Analysis object stores the types of measurements expected
       and the functions associated with each type of measurement
    """

    def __init__(self, measurements, functions, output_directory=None):
        """TODO: to be defined1.

        measurements: Dictionary with measurement names as keys
        functions: TODO

        """
        self._measurements = measurements
        self._functions = functions
        self._counter = 0
        if output_directory:
            dir_name = "results"
            dir_name += datetime.datetime.now().strftime('%I%M%W%d%Y%S')
            out_dir = join(output_directory, dir_name)
            self._output_directory = Path(out_dir)
            self._output_directory.mkdir()

    def analyze_measurement(self, measurement, data):
        results = dict()
        for m in self._measurements[measurement]:
            results[m] = self._functions[m](data)
        return results

    def dump_measurement(self, measurement, data, title='dump'):
        fname = '{0}-{1}.json'.format(title,  self._counter)
        path = join(self._output_directory, fname)
        results = dict()
        results[measurement] = self.analyze_measurement(measurement, data)
        with open(path, 'w') as f:
            dump(results, f)
        self._counter += 1
        return path

    @property
    def output_directory(self):
        return self._output_directory
