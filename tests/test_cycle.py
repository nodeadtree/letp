# Copyright (c) 2018 Juniper Overbeck
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import pytest
import numpy as np
import itertools as it
import letp as m

from pathlib import Path
# Test cases for the process_raw_data function. By default, these tests use no
# filter, but filtering functions can be passed as parameters to check their
# sanity.

# analysis output area


@pytest.fixture("session")
def analyzer_fixture(tmpdir_factory):
    path = tmpdir_factory.mktemp("analysis_testing_area")
    analyzer = m.Analyzer(measurements, analysis_functions, path)
    return analyzer


def analyzer_with_litter():
    path = Path("/tmp/analysis_test_area")
    analyzer = m.Analyzer(measurements, analysis_functions, path)
    return analyzer


# use this 'with pytest.raises(Exception):' for checking exceptions
size = [100, 100]
data_size = 100
data = [np.random.rand(*size) for i in range(data_size)]

measurements = {"individual_matrix": ["average"],
                "all_matrices": ["average_of_sum"]
                }

analysis_functions = {"average": np.average,
                      "average_of_sum": lambda x: np.average(np.sum(x))}

data_handler = m.DataHandler(data)

# Functions required to fill the arguments of Cycle


def step(x):
    return [("individual_matrix", x)]


def final(x):
    return [("all_matrices", [x[i] for i in x])]


@pytest.mark.parametrize("handler", [(data_handler)])
def test_single_evaluation_cycle(analyzer_fixture, handler):
    cycle = m.Cycle(analyzer_fixture, data_handler, step, final)
    assert cycle.run()


@pytest.mark.parametrize("handler", [(data_handler)])
def test_cycle_without_final_function(analyzer_fixture, handler):
    cycle = m.Cycle(analyzer_fixture, data_handler, step)
    assert cycle.run()

