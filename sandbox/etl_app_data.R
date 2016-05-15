library(dplyr)
library(ggplot2)
proj_dir = 'C:/Users/junchen/Documents/GitHub/pyMLC'

eng_app_data = read.csv(paste0(proj_dir,'/data/17zuoye/app_english.txt'),sep=',',header=T)
math_app_data = read.csv(paste0(proj_dir,'/data/17zuoye/app_math.txt'),sep=',',header=T)

# get rid of the users that appear once or twice

increase_density<-function (app_data){
    uid_eid_cnt = app_data %>% group_by(uid, category_id) %>% summarize(m=n()) %>% group_by(uid) %>% summarize(M=n()) 
    # if appears in less than 5 items, throw away. 40% users deleted but about 15% data deleted. 
    valid_uids = uid_eid_cnt %>% filter(M>=5) 
    retained_data = app_data %>% filter(uid %in% valid_uids$uid)
    return (retained_data)    
}

eng_retain_data = increase_density(eng_app_data)
math_retain_data = increase_density(math_app_data)

write.csv(math_retain_data, file=paste0(proj_dir,'/data/17zuoye/app_math_retain.csv'), row.names=F,quote=F)
write.csv(eng_retain_data, file=paste0(proj_dir,'/data/17zuoye/app_eng_retain.csv'), row.names=F,quote=F)

learning_curves = eng_retain_data %>% group_by(category_id, rank_idx) %>% summarize(pct=mean(atag))
qplot(data=learning_curves %>% filter(rank_idx<=15), x=rank_idx, y=pct, col=factor(category_id), geom='line')
