from collections import defaultdict

def data_loader_from_file(file_path, max_opportunity):
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
	user_result = defaultdict(dict)
	with open(file_path) as in_f:
		for line in in_f:
			if line == '\n':
				continue
			i_s, t_s, y_s = line.strip('\n').split(',')
			t = int(t_s)
			i = int(i_s)
			y = int(y_s)
			
			if t > max_opportunity:
				continue
			user_result[i][t] = y
			
	# run a second time
	response_data = []
	for user_log in user_result.values():
		user_response = [user_log[x] for x in sorted(user_log.keys())]
		response_data.append(user_response)

	return response_data

def data_loader_from_list(log_data, max_opportunity):
	# the log is a list of  (i,t,y) 
	user_result = defaultdict(dict)
	for log in log_data:
		i=log[0]; t=log[1]; y=log[2]
		if t > max_opportunity:
			continue
		user_result[i][t] = y		

	# run a second time
	response_data = []
	for user_log in user_result.values():
		user_response = [user_log[x] for x in sorted(user_log.keys())]
		response_data.append(user_response)

	return response_data
