import ijson


def stream_data(path_to_json_file, limit=0, start_at=0, path='questions'):
    i = 0
    if path == '':
        search_path = ''
    else:
        search_path = path + '.item'
    with open(path_to_json_file) as f:
        datareader = ijson.items(f, search_path)
        for record in datareader:
            i += 1
            if i < start_at + 1:
                continue
            if 0 < limit < i - start_at:
                return

            yield record


def get_ratio(a, b):
    if b == 0:
        return 0
    return a / b
