def data_loader(file_path, max_opportunity):
    '''
    # Input
    The data is comma delimited file with the following field
    user_id:
    practice_times: time index of the practice opportunities.
    1->N means the 1st, 2nd, ..., nth practice opportunity
    result: binary, 1 = right response

    # output
    [[Y1,Y2,...,Yt]]
    '''
    # load the dat
    user_result = {}
    with open(file_path) as in_f:
        for line in in_f:
            if line == '\n':
                continue
            uid, tid_s, result = line.strip('\n').split(',')
            tid = int(tid_s)
            if tid > max_opportunity:
                continue
            if uid not in user_result:
                user_result[uid] = {}
            user_result[uid][tid] = int(result)

    # run a second time
    response_data = []
    for user_log in user_result.itervalues():
        user_response = [user_log[x] for x in sorted(user_log.keys())]
        response_data.append(user_response)

    return response_data
