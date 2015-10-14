proj_dir = 'C:/Users/junchen/Documents/GitHub/pyMLC'
cnt_data = read.table(paste0(proj_dir, '/data/sample_app_data.txt'), sep=',',col.names=c('uid','tid','atag'))
library(dplyr)
library(ggplot2)

learning_curve = cnt_data %>% group_by(tid) %>% summarize(pct = 1-mean(atag), n=n()) %>% filter(tid<=15)
f1 = qplot(data=learning_curve, x=tid, y=pct, geom='line')
f2 = qplot(data=learning_curve, x=tid, y=n, geom='line')

