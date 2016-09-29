library(dplyr)
library(tidyr)
library(ggplot2)
library(gridExtra)

proj_dir = 'C:/Users/junchen/Documents/GitHub/pyMLC/data/bkt/test/'

params_effort = read.table(paste0(proj_dir,'effort_param_chain.txt'), sep=',')
params_no_effort = read.table(paste0(proj_dir,'constant_param_chain.txt'), sep=',')

sample_idx = seq(nrow(params_effort)/2,nrow(params_effort),10)

lrate_e = data.frame(q1=params_effort$V19[sample_idx],q2=params_effort$V20[sample_idx])
lrate_e = lrate_e %>% gather(qid,val)
m1 = qplot(data=lrate_e, x=val, col=factor(qid), geom='density')

lrate_ne = data.frame(q1=params_no_effort$V19[sample_idx],q2=params_no_effort$V20[sample_idx])
lrate_ne = lrate_ne %>% gather(qid,val)
m2=qplot(data=lrate_ne, x=val, col=factor(qid), geom='density')


grid.arrange(m2,m1)



