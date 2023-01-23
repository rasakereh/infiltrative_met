setwd('/Users/rasakereh/Desktop/uni/infiltrative_met/src')

require(dplyr)
require(ggplot2)
require(stringr)
require(caret)
require(infotheo)
require(umap)
require(reshape2)
require(limma)

cluster.cnt <- 3

features <- read.csv('reshaped_dual.csv')
grades <- features %>% select(case, domin_s, d_per, high_s, h_per, total)
features <- features %>% select(-domin_s, -d_per, -high_s, -h_per, -total)

feature.names <- colnames(features) %>% str_split('_') %>% sapply(function(exploded){
  exploded %>% head(-1) %>% Reduce(f=function(a, b){paste(a, b, sep='_')})
}) %>% unlist %>% unique

no.var.cols <- colnames(features)[features %>% apply(2, sd) == 0]
features <- features %>% select(-all_of(no.var.cols)) %>% select(-ends_with('_count'))

pdf('analysis/data_overview.pdf')
{
  anonym.feats <- features %>% select(-case) %>% t() %>% as.data.frame()
  colnames(anonym.feats) <- paste0('case', features$case)
  boxplot(anonym.feats, main="Feature distribution per case")
  
  log.feats <- log(anonym.feats - min(anonym.feats) + .01)
  boxplot(log.feats, main="Log-transformed features")
  
  log.feats.norm <- normalizeQuantiles(log.feats)
  boxplot(log.feats.norm, main="Log-transformed + quantile-normalized features")
  
  log.features <- log.feats %>% t %>% as.data.frame %>% mutate(case = features$case, .before=everything())
  rownames(log.features) <- NULL
  log.norm.features <- log.feats.norm %>% t %>% as.data.frame %>% mutate(case = features$case, .before=everything())
  rownames(log.norm.features) <- NULL
  
  for(feature.name in feature.names){
    sub.data <- features %>% select(case, contains(feature.name))
    melted.sub <- melt(sub.data, id=c('case'))
    
    img_data_corr <- ggplot(melted.sub, aes(x=value)) +
      geom_density() +
      facet_wrap(~variable, scale='free') + 
      ggtitle(paste0('Distribution of ', feature.name)) + theme_bw()
    print(img_data_corr)
  }
  
  for(feature.name in feature.names){
    sub.data <- log.features %>% select(case, contains(feature.name))
    melted.sub <- melt(sub.data, id=c('case'))
    
    img_data_corr <- ggplot(melted.sub, aes(x=value)) +
      geom_density() +
      facet_wrap(~variable, scale='free') + 
      ggtitle(paste0('Distribution of log-transformed ', feature.name)) + theme_bw()
    print(img_data_corr)
  }
  
  for(feature.name in feature.names){
    sub.data <- log.norm.features %>% select(case, contains(feature.name))
    melted.sub <- melt(sub.data, id=c('case'))

    img_data_corr <- ggplot(melted.sub, aes(x=value)) +
      geom_density() +
      facet_wrap(~variable, scale='free') +
      ggtitle(paste0('Distribution of transformed + normalized', feature.name)) + theme_bw()
    print(img_data_corr)
  }
}
dev.off()


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
  sub.data$grade <- cut(grades$total, quantile(grades$total, (0:cluster.cnt)/cluster.cnt), include.lowest=T, right=F) %>% as.integer
  sub.data <- sub.data %>% filter(grade != 2) %>% mutate(grade = as.factor(grade))
  
  img_data_corr <- ggplot(sub.data, aes(x=!!sym(feature.name), fill=grade)) +
    geom_density(alpha=.7)
  print(img_data_corr)
}
dev.off()

pdf('analysis/data_hist_all.pdf')
for(feature.name in feature.names){
  sub.data <- feature.pcs %>% select(case, all_of(feature.name))
  sub.data$grade <- cut(grades$total, quantile(grades$total, (0:cluster.cnt)/cluster.cnt), include.lowest=T, right=F) %>% as.integer
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
  invasive = cut(grades$total, quantile(grades$total, (0:cluster.cnt)/cluster.cnt), include.lowest=T, right=F) %>% as.integer %>% as.factor()#ifelse(grades$domin_s == 1, 'minimal', ifelse(grades$domin_s == 2, 'moderate', 'high')) %>% as.factor() # cut(grades$total, quantile(grades$total, (0:3)/3)), include.lowest=T) %>% as.integer %>% as.factor()
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


{
  groupings <- cut(grades$total, quantile(grades$total, (0:cluster.cnt)/cluster.cnt), include.lowest=T, right=F) %>% as.integer
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
}


{
  groupings <- cut(grades$total, quantile(grades$total, (0:cluster.cnt)/cluster.cnt), include.lowest=T, right=F) %>% as.integer
  low.invasion <- features %>% filter(groupings == 1)
  high.invasion <- features %>% filter(groupings == 3)
  low.params <- low.invasion %>% apply(2, function(col){c(m=mean(col), s=sd(col))})
  high.params <- high.invasion %>% apply(2, function(col){c(m=mean(col), s=sd(col))})
  
  low.z <- abs((low.invasion - low.params[rep('m', nrow(low.invasion)),]) / low.params[rep('s', nrow(low.invasion)),])
  high.z <- abs((high.invasion - high.params[rep('m', nrow(high.invasion)),]) / high.params[rep('s', nrow(high.invasion)),])
  
  sus.lows <- which(low.z > 2, arr.ind = T) %>% as.data.frame %>%
    filter(col != 1) %>% group_by(row) %>%
    summarise(n = n()) %>% mutate(n = n/(ncol(low.lz) - 1))
  sus.highs <- which(high.z > 2 , arr.ind = T) %>% as.data.frame %>%
    filter(col != 1) %>% group_by(row) %>%
    summarise(n = n()) %>% mutate(n = n/(ncol(high.hz) - 1))
  
  outliers <- rbind.data.frame(
    sus.highs %>% mutate(case = high.invasion$case[sus.highs$row], type = 'high'),# %>% filter(n > .5),
    sus.lows %>% mutate(case = low.invasion$case[sus.lows$row], type = 'low')# %>% filter(n > .5)
  ) %>% inner_join(grades, by = c(case='case'))
}


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





feature.files <- list.files(path='steps', pattern='*.csv')
indiv.features = lapply(feature.files, function(file.name){
  case <- str_split(file.name, '_')[[1]][1] %>% substr(3, 5) %>% as.integer()
  csv.file <- read.csv(paste0('steps/', file.name))
  csv.file <- csv.file %>% mutate(case = case, part = 0:(n()-1), .before=everything())
  csv.file[,3:ncol(csv.file)] <- log(csv.file[,3:ncol(csv.file)] + .01)
  
  csv.file
})
indiv.features <- Reduce(rbind.data.frame, indiv.features)


{
  feature.means <- apply(indiv.features, 2, mean)
  dim(feature.means) <- c(1, length(feature.means))
  feature.comp <- indiv.features > feature.means[rep(1, nrow(indiv.features)),]
  
  correlation <- data.frame(
    'invasion_count'=1,
    'brain_compactness'=-1,
    'brain_convexity'=-1,
    'brain_solidity'=-1,
    'convex_overlap'=1,
    'filled_overlap'=1
  )
  
  is.invasive.decision <- !xor(
    (correlation > 0)[rep(1, nrow(feature.comp)),],
    feature.comp[,3:ncol(feature.means)]
  )
  
  is.invasive.vote <- is.invasive.decision %>% as.data.frame
  pca.vote <- prcomp(indiv.features %>% select(-case, -part))$x %>%
    as.data.frame() %>% mutate(case = indiv.features$case) %>%
    filter(indiv.features$part < 30) %>% inner_join(grades, by='case')
  ggplot(pca.vote, aes(PC1, PC2, col=as.factor(total))) + geom_point(size=3)
  
  is.invasive.vote$vote <- is.invasive.vote %>% apply(1, mean)
  is.invasive.vote <- is.invasive.vote %>%
    mutate(case = indiv.features$case, part = indiv.features$part, .before=everything())
  
  plate.intuition <- is.invasive.vote %>% group_by(case) %>%
    summarise(min.part = (which.min(vote)-1), min = min(vote), max.part = (which.max(vote)-1), max = max(vote))
  
  plate.votes <- is.invasive.vote %>%
    group_by(case) %>% mutate(weight = (n() - part) / (sum(part) + n())) %>%
    ungroup() %>% group_by(case, vote) %>%
    summarise(density = sum(weight)) %>% dcast(case ~ vote, value.var = 'density', fill=0)
  colnames(plate.votes) <- substr(colnames(plate.votes), 0, 4)
  
  plate.guess <- data.frame(
    case = plate.votes$case,
    minimal = plate.votes$`0` + plate.votes$`0.16` + plate.votes$`0.33` + plate.votes$`0.5`,
    high = plate.votes$`0.66` + plate.votes$`0.83` + plate.votes$`1`,
    invasive = grades$domin_s %>% as.factor()
  )
  
  ggplot(plate.guess, aes(minimal, high, col=invasive, label=case)) +
    geom_point(size=3) + geom_text(hjust=2, vjust=0)
  
  inter.non.conform <- apply(is.invasive.decision, 2, function(feature.col){
    xor(feature.col, is.invasive.decision) %>% apply(1, sum)
  }) %>% as.data.frame() %>%
    mutate(case = indiv.features$case, .before=everything()) %>%
    group_by(case) %>% summarise_all(mean)
  
  binary.invasion <- indiv.features %>% inner_join(grades, by='case') %>%
    select(total) %>% mutate(bin.invasion = total < mean(total)) %>%
    select(bin.invasion)
  
  pca.vote$total < mean(pca.vote$total)
  
  truth.non.conform <- apply(is.invasive.decision, 2, function(feature.col){
    xor(feature.col, binary.invasion$bin.invasion)
  }) %>% as.data.frame() %>%
    mutate(case = indiv.features$case, .before=everything()) %>%
    group_by(case) %>% summarise_all(mean) %>% 
    inner_join(grades, by='case') %>% select(-d_per, -h_per, -total)
  truth.non.conform <- truth.non.conform %>% melt(id=c('case', 'domin_s', 'high_s'))
  truth.non.conform$high_s[is.na(truth.non.conform$high_s)] <- truth.non.conform$domin_s[is.na(truth.non.conform$high_s)]
  
  ggplot(truth.non.conform, aes(variable, value, label=case, col=as.factor(domin_s))) +
    geom_point(size=3) + geom_text(hjust=2, vjust=0)
  ggplot(truth.non.conform, aes(variable, value, label=case, col=as.factor(high_s))) +
    geom_point(size=3) + geom_text(hjust=2, vjust=0)
  
  case.misconception <- truth.non.conform %>% select(-variable) %>%
    group_by(case) %>% summarise_all(mean) %>% arrange(-value)
}

