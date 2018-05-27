from multiprocessing import Pool


def process_raw_data(data, reader=lambda x: x,
                     processor=lambda x: x):
    assert data is not None
    return processor(reader(data))


def pre_analysis_stage(data, wrapper_instructions):
    assert data is not None
    for k in wrapper_instructions:
        data = k(data)
    return data


def model_testing(data,  test_schema, test_unit, analytics):
    with Pool() as p:
        results = p.map(test_unit, test_schema(data))
    return analytics(results)
