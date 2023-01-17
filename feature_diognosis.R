setwd('/Users/rasakereh/Desktop/uni/infiltrative_met/src')

require(dplyr)
require(ggplot2)
require(stringr)
require(caret)
require(infotheo)
require(umap)
require(reshape2)

cluster.cnt <- 3

features <- read.csv('reshaped_dual.csv')
grades <- features %>% select(case, domin_s, d_per, high_s, h_per, total)
features <- features %>% select(-domin_s, -d_per, -high_s, -h_per, -total)

feature.names <- colnames(features) %>% str_split('_') %>% sapply(function(exploded){
  exploded %>% head(-1) %>% Reduce(f=function(a, b){paste(a, b, sep='_')})
}) %>% unlist %>% unique

no.var.cols <- colnames(features)[features %>% apply(2, sd) == 0]
features <- features %>% select(-all_of(no.var.cols))


pdf('analysis/data_corr.pdf')
for(feature.name in feature.names){
  sub.data <- features %>% select(case, contains(feature.name))
  sub.data$grade <- grades$total
  melted.sub <- melt(sub.data, id=c('case', 'grade'))
  
  img_data_corr <- ggplot(melted.sub, aes(x=grade, y=value)) +
    geom_point(size=3) +
    geom_smooth(method = "lm") +
    facet_wrap(~variable, scale='free')
  print(img_data_corr)
}
dev.off()


feature.pcs <- feature.names %>% lapply(function(feature.name){
  sub.data <- features %>% select(case, contains(feature.name))
  pca_res <- prcomp(sub.data %>% select(-case), scale=T)$x
  pca_res[,'PC1']
}) %>% cbind.data.frame
colnames(feature.pcs) <- feature.names
feature.pcs$case <- features$case


pdf('analysis/data_hist.pdf')
for(feature.name in feature.names){
  sub.data <- feature.pcs %>% select(case, all_of(feature.name))
  sub.data$grade <- cut(grades$total, quantile(grades$total, (0:cluster.cnt)/cluster.cnt), include.lowest=T) %>% as.integer
  sub.data <- sub.data %>% filter(grade != 2) %>% mutate(grade = as.factor(grade))
  
  img_data_corr <- ggplot(sub.data, aes(x=!!sym(feature.name), fill=grade)) +
    geom_density(alpha=.7)
  print(img_data_corr)
}
dev.off()

pdf('analysis/data_hist_all.pdf')
for(feature.name in feature.names){
  sub.data <- feature.pcs %>% select(case, all_of(feature.name))
  sub.data$grade <- cut(grades$total, quantile(grades$total, (0:cluster.cnt)/cluster.cnt), include.lowest=T) %>% as.integer
  sub.data <- sub.data %>% mutate(grade = as.factor(grade))
  
  img_data_corr <- ggplot(sub.data, aes(x=!!sym(feature.name), fill=grade)) +
    geom_density(alpha=.7)
  print(img_data_corr)
}
dev.off()


pca_res <- prcomp(features %>% select(-case), scale=T)$x
pca_feat <- data.frame(
  case = features$case,
  pca_res,
  invasive = cut(grades$total, quantile(grades$total, (0:cluster.cnt)/cluster.cnt), include.lowest=T) %>% as.integer %>% as.factor()#ifelse(grades$domin_s == 1, 'minimal', ifelse(grades$domin_s == 2, 'moderate', 'high')) %>% as.factor() # cut(grades$total, quantile(grades$total, (0:3)/3)), include.lowest=T) %>% as.integer %>% as.factor()
)

{
pdf('analysis/data_dim_red.pdf')

umap_feat <- umap(features %>% select(-case))$layout %>% as.data.frame()
umap_feat <- cbind.data.frame(umap_feat, invasive=pca_feat$invasive)
img_data_corr <- ggplot(umap_feat, aes(V1, V2, col=invasive)) + geom_point(size=5) +
  ggtitle('UMAP all features')
print(img_data_corr)

umap_feat <- umap(feature.pcs %>% select(-case))$layout %>% as.data.frame()
umap_feat <- cbind.data.frame(umap_feat, invasive=pca_feat$invasive)
img_data_corr <- ggplot(umap_feat, aes(V1, V2, col=invasive)) + geom_point(size=5) + 
  ggtitle('UMAP PCA features')
print(img_data_corr)

img_data_corr <- ggplot(pca_feat, aes(x=PC1, fill=invasive)) +
  geom_density(alpha=.7)
print(img_data_corr)

img_data_corr <- ggplot(pca_feat, aes(x=PC2, fill=invasive)) +
  geom_density(alpha=.7)
print(img_data_corr)

dev.off()
}



groupings <- cut(grades$total, quantile(grades$total, (0:cluster.cnt)/cluster.cnt), include.lowest=T) %>% as.integer
low.invasion <- features %>% filter(groupings == 1)
high.invasion <- features %>% filter(groupings == 3)
low.params <- low.invasion %>% apply(2, function(col){c(m=mean(col), s=sd(col))})
high.params <- high.invasion %>% apply(2, function(col){c(m=mean(col), s=sd(col))})

low.lz <- abs((low.invasion - low.params[rep('m', nrow(low.invasion)),]) / low.params[rep('s', nrow(low.invasion)),])
low.hz <- abs((low.invasion - high.params[rep('m', nrow(low.invasion)),]) / high.params[rep('s', nrow(low.invasion)),])
high.lz <- abs((high.invasion - low.params[rep('m', nrow(high.invasion)),]) / low.params[rep('s', nrow(high.invasion)),])
high.hz <- abs((high.invasion - high.params[rep('m', nrow(high.invasion)),]) / high.params[rep('s', nrow(high.invasion)),])

sus.lows <- which(low.lz > low.hz, arr.ind = T) %>% as.data.frame %>%
  filter(col != 1) %>% group_by(row) %>%
  summarise(n = n()) %>% mutate(n = n/(ncol(low.lz) - 1))
sus.highs <- which(high.lz < high.hz, arr.ind = T) %>% as.data.frame %>%
  filter(col != 1) %>% group_by(row) %>%
  summarise(n = n()) %>% mutate(n = n/(ncol(high.hz) - 1))

candids <- rbind.data.frame(
  sus.highs %>% mutate(case = high.invasion$case[sus.highs$row], type = 'high') %>% filter(n > .5),
  sus.lows %>% mutate(case = low.invasion$case[sus.lows$row], type = 'low') %>% filter(n > .5)
) %>% inner_join(grades, by = c(case='case'))


pdf('analysis/sus_candidates.pdf')
for(feature.name in feature.names){
  sub.data <- rbind.data.frame(
    low.invasion %>% select(case, contains(feature.name)) %>% mutate(group='low'),
    high.invasion %>% select(case, contains(feature.name)) %>% mutate(group='high')
  )
  sub.data$group <- as.factor(sub.data$group)
  
  sub.data.melted <- melt(sub.data, id=c('case', 'group'))
  sub.data.melted$variable <- as.factor(sub.data.melted$variable)
  
  img_data_corr <- ggplot(sub.data.melted, aes(x=variable, y=value, col=group, label=case)) +
    geom_point(size=3) + geom_text(hjust=2, vjust=0)
  print(img_data_corr)
}
dev.off()

pdf('analysis/sus_candidates_aggr.pdf')
for(feature.name in feature.names){
  sub.data <- rbind.data.frame(
    feature.pcs %>% filter(groupings == 1) %>%
      select(case, contains(feature.name)) %>% mutate(group='low'),
    feature.pcs %>% filter(groupings == 3) %>%
      select(case, contains(feature.name)) %>% mutate(group='high')
  )
  sub.data$group <- as.factor(sub.data$group)
  
  sub.data.melted <- melt(sub.data, id=c('case', 'group'))
  sub.data.melted$variable <- as.factor(sub.data.melted$variable)
  
  img_data_corr <- ggplot(sub.data.melted, aes(x=paste(variable, group), y=value, col=group, label=case)) +
    geom_point(size=3) + geom_text(hjust=2, vjust=0)
  print(img_data_corr)
}
dev.off()
