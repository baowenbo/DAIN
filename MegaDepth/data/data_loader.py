
def CreateDataLoader(_root, _list_dir, _input_height, _input_width, is_flip = True, shuffle =  True):
    data_loader = None
    from data.aligned_data_loader import AlignedDataLoader
    data_loader = AlignedDataLoader(_root, _list_dir, _input_height, _input_width, is_flip, shuffle)
    return data_loader

def CreateDataLoader_TEST(_root, _list_dir, _input_height, _input_width):
    data_loader = None
    from data.aligned_data_loader import AlignedDataLoader_TEST
    data_loader = AlignedDataLoader_TEST(_root, _list_dir, _input_height, _input_width)

    return data_loader
