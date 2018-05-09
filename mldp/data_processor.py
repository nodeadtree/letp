def process_data(data, reader=lambda x: x,
                 processor=lambda x: x):
    assert data is not None
    return reader(processor(data))


def wrapper_stage(data, wrapper_instructions):
    assert data is not None
    for k in wrapper_instructions:
        data = k(data)
    return data
