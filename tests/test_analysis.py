# Copyright (c) 2018 Juniper Overbeck
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import pytest
import itertools as it
import numpy as np
import mldp.analysis as a

from json import load


def generate_test_matrix(size, random=True):
    if random:
        return np.random.rand(*size)
    else:
        return np.ones(*size)


def identity(x):
    return x


# Test cases for the process_raw_data function. By default, these tests use no
# filter, but filtering functions can be passed as parameters to check their
# sanity.

# use this 'with pytest.raises(Exception):' for checking exceptions

default_test_sizes = [1, 10, 1000]
size_list = [(i, k) for i, k in it.product(default_test_sizes,
             default_test_sizes[1:])]


measurements_dict = {"default": ["identity1", "identity2"]}
functions_dict = {"identity1": identity,
                  "identity2": identity}


test_params = [(measurements_dict, functions_dict, i) for i in size_list]


@pytest.fixture(scope="session")
def test_dir(tmpdir_factory):
    path = tmpdir_factory.mktemp("analysis_test_area")
    return path


@pytest.mark.parametrize("measurements,functions,test_size", test_params)
def test_analyze_measurement_without_directory(measurements,
                                               functions, test_size):
    analysis_object = a.Analyzer(measurements, functions)
    test_measurement = np.random.rand(*test_size)
    result = analysis_object.analyze_measurement("default", test_measurement)
    for i in result:
        assert test_measurement is result[i]


def test_analyze_instantiation_with_directory(test_dir):
    analysis_object = a.Analyzer(measurements_dict, functions_dict, test_dir)
    analysis_object.output_directory.rmdir()


@pytest.mark.parametrize("measurement, test_size", [("default", (10, 100))])
def test_dump_measurement(test_dir, measurement, test_size):
    analysis_object = a.Analyzer(measurements_dict, functions_dict, test_dir)
    for k in range(10):
        data = np.random.rand(*test_size)
        path = analysis_object.dump_measurement(measurement, data)
        with open(path, 'r') as f:
            data = load(f)
            assert measurement in data
