
# Copyright (c) 2018 Juniper Overbeck
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import pytest
import numpy as np
import itertools as it
import letp.data_handler as a

# Test cases for the process_raw_data function. By default, these tests use no
# filter, but filtering functions can be passed as parameters to check their
# sanity.

# use this 'with pytest.raises(Exception):' for checking exceptions
sizes = [1, 10]
data_sizes = [i for i in it.product(sizes, sizes)]
test_sizes = sizes


@pytest.mark.parametrize("data_size, test_size",
                         [i for i in it.product(data_sizes, test_sizes)])
def test_data_handler(data_size, test_size):
    data = [np.random.rand(*data_size) for i in range(test_size)]
    handler = a.DataHandler(data)
    for i, j in zip(handler, data):
        assert i is j


@pytest.mark.parametrize("data_size, test_size",
                         [i for i in it.product(data_sizes, test_sizes[1:])])
def test_add_data(data_size, test_size):
    data = [np.random.rand(*data_size) for i in range(test_size)]
    handler = a.DataHandler([data[0]])
    n_data = data[1:]
    for i in n_data:
        handler.add_data(i)
    for i, j in zip(handler, data):
        assert i is j
