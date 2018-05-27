import pytest
import itertools as it
import numpy as np
import mldp


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
default_test_sizes = [1, 10, 1000]
size_combos = [(i, k) for i, k in it.product(default_test_sizes,
               default_test_sizes[1:])]
size_col = [size_combos]
raw_data_functions = [identity]
read_data_functions = [identity]
test_process_raw_data_params = [(size_combos, identity, identity)]


@pytest.mark.parametrize("reader,processor", zip(raw_data_functions,
                                                 raw_data_functions))
def test_None_input_failure(reader, processor):
    with pytest.raises(Exception):
        mldp.process_raw_data(None, reader, processor)


@pytest.mark.parametrize("reader,processor", zip(raw_data_functions,
                                                 raw_data_functions))
def test_no_input_failure(reader, processor):
    with pytest.raises(Exception):
        mldp.process_raw_data(reader=reader, processor=processor)


@pytest.mark.parametrize("sizes,reader,processor", zip(size_col,
                                                       raw_data_functions,
                                                       raw_data_functions))
def test_process_raw_data_for_numpy_arrays(sizes, reader, processor):
    for i in sizes:
        assert isinstance(mldp.process_raw_data(generate_test_matrix(i),
                                                reader, processor), np.ndarray)


@pytest.mark.parametrize("sizes,reader,processor", zip(size_col,
                                                       raw_data_functions,
                                                       raw_data_functions))
def test_process_raw_data_for_numpy_array_dimension(sizes, reader, processor):
    print(sizes)
    for i in sizes:
        assert mldp.process_raw_data(generate_test_matrix(i),
                                     reader, processor).ndim > 1


test_wrapper_instructions = [[identity]]


@pytest.mark.parametrize("wrapper_instructions",
                         test_wrapper_instructions)
def test_pre_analysis_stage_no_data(wrapper_instructions):
    with pytest.raises(Exception):
        mldp.pre_analysis_stage(None, wrapper_instructions)


@pytest.mark.parametrize("sizes,wrapper_instructions",
                         zip(size_col, test_wrapper_instructions))
def test_pre_analysis_stage_for_output(sizes, wrapper_instructions):
    for i in sizes:
        assert mldp.pre_analysis_stage(generate_test_matrix(i),
                                       wrapper_instructions) is not None


@pytest.mark.parametrize("sizes,wrapper_instructions",
                         zip(size_col, test_wrapper_instructions))
def test_pre_analysis_stage_for_numpy_arrays(sizes, wrapper_instructions):
    for i in sizes:
        assert isinstance(mldp.pre_analysis_stage(generate_test_matrix(i),
                                                  wrapper_instructions),
                          np.ndarray)


@pytest.mark.parametrize("sizes,wrapper_instructions",
                         zip(size_col, test_wrapper_instructions))
def test_pre_analysis_stage_for_array_dimension(sizes, wrapper_instructions):
    for i in sizes:
        assert mldp.pre_analysis_stage(generate_test_matrix(i),
                                       wrapper_instructions).ndim > 1


# Composition testing
param_string = "sizes,reader,processor,wrapper_instructions"


@pytest.mark.parametrize(param_string, zip(size_col, read_data_functions,
                                           raw_data_functions,
                                           test_wrapper_instructions))
def test_process_wrapper_composition(sizes, reader, processor,
                                     wrapper_instructions):
    for i in sizes:
        mldp.pre_analysis_stage(mldp.process_raw_data(generate_test_matrix(i),
                                                      reader, processor),
                                wrapper_instructions)
    pass
