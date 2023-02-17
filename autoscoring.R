setwd('/Users/rasakereh/Desktop/uni/infiltrative_met/src')

require(dplyr)
require(ggplot2)
require(stringr)
require(caret)
require(infotheo)
require(reshape2)
require(limma)
require(GGally)
require(fpc)
require(mclust)
require(umap)
require(randomForest)
require(NMF)


z.normalization <- function(dataset){
  col.n <- colnames(dataset)
  row.n <- rownames(dataset)
  
  z.normed <- dataset %>% t %>% apply(2, scale) %>% t
  colnames(z.normed) <- col.n
  rownames(z.normed) <- row.n
  
  z.normed
}

grades <- read.csv('reshaped_dual.csv') %>% select(case, domin_s, d_per, high_s, h_per, total)

feature.files <- list.files(path='steps', pattern='*.csv')
indiv.features <- lapply(feature.files, function(file.name){
  features.to.exclude <- c('delaunay_score_max')
  case <- str_split(file.name, '_')[[1]][1] %>% substr(3, 5) %>% as.integer()
  csv.file <- read.csv(paste0('steps/', file.name))
  csv.file <- csv.file %>% mutate(case = case, part = 0:(n()-1), .before=everything())
  csv.file[,5:ncol(csv.file)] <- sqrt(csv.file[,5:ncol(csv.file)])
  
  csv.file %>% select(-all_of(features.to.exclude))
})
indiv.features <- Reduce(rbind.data.frame, indiv.features)

############### TODO, IMPOOOOOOORTTAAAANNNT
indiv.features[is.na(indiv.features)] <- 0
############### TODO, IMPOOOOOOORTTAAAANNNT

indiv.features[5:ncol(indiv.features)] <- indiv.features[5:ncol(indiv.features)] %>%
  t %>% z.normalization %>% normalizeQuantiles %>% t



micro.scores <- read.csv('micro_scores.csv')

################################################################################
my.grades <- indiv.features %>% select(case, part, whole_brain_area, whole_tumor_area) %>%
  mutate(score = micro.scores$new.truth, interface = sqrt(mapply(min, whole_brain_area, whole_tumor_area))) %>%
  group_by(case) %>% mutate(percent = interface / sum(interface)) %>% ungroup %>%
  group_by(case, score) %>% summarise(total_area = sum(percent)) %>%
  dcast(case ~ score, value.var = 'total_area', fill=0)
my.grades <- my.grades %>%
  mutate(
    domin_s = apply(my.grades, 1, function(row){which.max(row[-1])}),
    high_s = apply(my.grades, 1, function(row){max((1:3)[row[-1] > .1])})
  )

my.grades <- my.grades %>% mutate(
  d_per = sapply(1:nrow(my.grades), function(i){my.grades[i, 1+my.grades$domin_s[i]]}),
  h_per = sapply(1:nrow(my.grades), function(i){my.grades[i, 1+my.grades$high_s[i]]})
) %>% mutate(
  total = ifelse(domin_s == high_s, d_per * domin_s, d_per * domin_s + h_per * high_s)
) %>% select(-`1`, -`2`, -`3`)

################################################################################
melted.ind.feat <- indiv.features %>% select(-whole_brain_area, -whole_tumor_area) %>%
  mutate(score=as.factor(micro.scores$truth)) %>%
  melt(id=c('case', 'part', 'score')) %>% select(-case, -part)
ggplot(melted.ind.feat, aes(x=value, fill=score)) +
  geom_density(alpha=.7) + facet_wrap(~variable)
feature.info <- indiv.features %>% select(-case, -part, -whole_brain_area, -whole_tumor_area) %>% mutate_all(function(col){
  mutinformation(
    col %>% discretize,
    micro.scores$new.truth %>% as.factor
  )
}) %>% summarise_all(mean)

mapping <- data.frame(
  'invasion_count_cnt'=1:3,
  'invasion_count_min_area'=c(1,3,2),
  'invasion_count_max_area'=c(1,3,2),
  'invasion_count_med_area'=c(1,3,2),
  'delaunay_score_min'=3:1,
  'delaunay_score_quartile1'=3:1,
  'delaunay_score_median'=3:1,
  'delaunay_score_quartile3'=3:1,
  'tumor_components_cnt'=1:3,
  'tumor_components_min_area'=c(1,3,2),
  'tumor_components_max_area'=c(1,3,2),
  'tumor_components_med_area'=c(1,3,2),
  'interface_ellipse_eccent_quants_0.1'=c(2,3,1),
  'interface_ellipse_eccent_quants_0.5'=c(3,1,2),
  'interface_ellipse_eccent_quants_0.9'=c(2,3,1),
  'interface_ellipse_area_quants_0.1'=3:1,
  'interface_ellipse_area_quants_0.5'=c(1,3,2),
  'interface_ellipse_area_quants_0.9'=c(1,3,2),
  'cooccurrence_mat_0'=c(1,3,2),
  'cooccurrence_mat_1'=1:3,
  'cooccurrence_mat_2'=1:3,
  'cooccurrence_mat_3'=c(2,1,3),
  'cooccurrence_mat_4'=c(2,1,3),
  'cooccurrence_mat_5'=1:3,
  'cooccurrence_mat_6'=3:1,
  'brain_compactness'=3:1,
  'brain_convexity'=3:1,
  'brain_solidity'=3:1,
  'convex_overlap'=c(1,3,2),
  'filled_overlap_cnt'=1:3,
  'filled_overlap_min_area'=c(1,3,2),
  'filled_overlap_max_area'=c(1,3,2),
  'filled_overlap_med_area'=c(1,3,2)
)

feature.families <- list(1:4, 5:8, 9:12, 13:18, 19:25, 26:28, 30:33)
feature.families %>% lapply(function(feat.range){
  feat.names <- Reduce(paste, colnames(mapping)[feat.range])
  family.pca <- indiv.features %>%
    select(-case, -part, -whole_brain_area, -whole_tumor_area) %>% 
    select(feat.range) %>% prcomp()
  
  curr.plot <- ggplot(
    family.pca$x %>% as.data.frame(),
    aes(PC1, PC2, col=as.factor(micro.scores$new.truth)),
  ) + geom_point(size=5) + ggtitle(feat.names)
  
  print(curr.plot)
  
  NULL
})

feature.families %>% lapply(function(feat.range){
  ggpairs(
    indiv.features %>% select(-case, -part, -whole_brain_area, -whole_tumor_area),
    columns=feat.range,
    mapping=aes(col=as.factor(micro.scores$new.truth), alpha=.7)
  ) %>% print()
})

feature.groups <- colnames(indiv.features)[5:ncol(indiv.features)] %>% sapply(function(feat.name){
  curr.feature <- indiv.features[, feat.name]
  quantiles <- quantile(curr.feature, (0:3)/3)
  quantiles[1] <- quantiles[1] - 1e-2
  quantiles[length(quantiles)] <- quantiles[length(quantiles)] + 1e-2
  curr.groups <- cut(curr.feature, quantiles, include.lowest=T) %>% as.integer
  curr.groups <- mapping[curr.groups, feat.name]
  
  #############
  # gmm.cuts <- Mclust(curr.feature, G = 3, modelNames='V')
  # gmm.groups <- gmm.cuts$classification
  #############
  
  curr.groups
}) %>% as.data.frame()

conform.lbl <- feature.groups %>% mutate_all(function(col){
  col == micro.scores$new.truth
}) %>% summarise_all(mean)

indiv.features %>%
  select(-whole_brain_area, -whole_tumor_area, -case, -part) %>%
  mutate(score = micro.scores$new.truth, .before=everything()) %>%
  write.csv('dnn_data.csv', row.names = F)

pca.res <- indiv.features %>%
  select(-case, -part, -whole_brain_area, -whole_tumor_area) %>% prcomp

mutinformation(
  pca.res$x %>% as.data.frame %>% select(PC1, PC2, PC3, PC4, PC5) %>% discretize,
  micro.scores$new.truth %>% as.factor
)
mutinformation(
  pca.res$x %>% discretize,
  micro.scores$new.truth %>% as.factor
)

vote.df <- feature.groups %>% apply(1, function(row){
  which.max(table(factor(row, levels = 1:3)))
  # round(mean(row))
  # table(factor(row, levels = 1:3)) / length(row)
})

patch.conf.mat <- confusionMatrix(
  table(
    as.factor(vote.df),
    as.factor(micro.scores$new.truth)
  ),
  mode='prec_recall'
)

pca.vote <- pca.res$x %>% as.data.frame() %>%
  mutate(
    case = indiv.features$case,
    score=micro.scores$new.truth,
    estimate=vote.df,
    interface = sqrt(mapply(min, indiv.features$whole_brain_area, indiv.features$whole_tumor_area))
)

ggplot(pca.vote, aes(PC1, PC2, col=as.factor(score))) + geom_point(size=3)
ggplot(pca.vote, aes(PC1, PC2, col=as.factor(estimate))) + geom_point(size=3)

umap_feat <- umap(indiv.features %>%
                    select(-case, -part, -whole_brain_area, -whole_tumor_area))$layout %>%
  as.data.frame()
umap_feat <- cbind.data.frame(umap_feat, score=factor(micro.scores$new.truth), estimate=factor(vote.df))
ggplot(umap_feat, aes(V1, V2, col=score)) + geom_point(size=3)
ggplot(umap_feat, aes(V1, V2, col=estimate)) + geom_point(size=3)

ggpairs(pca.vote, columns=1:5, mapping=aes(col=as.factor(micro.scores$new.truth), alpha=.7))


final.decision <- indiv.features %>% select(case, part, whole_brain_area, whole_tumor_area) %>%
  mutate(score = vote.df, interface = sqrt(mapply(min, whole_brain_area, whole_tumor_area))) %>%
  group_by(case) %>% mutate(percent = interface / sum(interface)) %>% ungroup %>%
  group_by(case, score) %>% summarise(total_area = sum(percent)) %>%
  dcast(case ~ score, value.var = 'total_area', fill=0)
final.decision <- final.decision %>%
  mutate(
    domin_s = apply(final.decision, 1, function(row){which.max(row[-1])}),
    high_s = apply(final.decision, 1, function(row){max((1:3)[row[-1] > .15])})
  )

plate.conf.mat.d <- confusionMatrix(
  table(
    factor(final.decision$domin_s, levels=1:3),
    factor(my.grades$domin_s, levels=1:3)
  ),
  mode='prec_recall'
)

plate.conf.mat.h <- confusionMatrix(
  table(
    factor(final.decision$high_s, levels=1:3),
    factor(my.grades$high_s, levels=1:3)
  ),
  mode='prec_recall'
)

######################################
#ABI: Area, Border, Intensity
ABI_groups <- c(
  'invasion_count_cnt'='I',
  'invasion_count_min_area'='A',
  'invasion_count_max_area'='A',
  'invasion_count_med_area'='A',
  'delaunay_score_min'='I',
  'delaunay_score_quartile1'='I',
  'delaunay_score_median'='I',
  'delaunay_score_quartile3'='I',
  'tumor_components_cnt'='I',
  'tumor_components_min_area'='A',
  'tumor_components_max_area'='A',
  'tumor_components_med_area'='A',
  'interface_ellipse_eccent_quants_0.1'='B',
  'interface_ellipse_eccent_quants_0.5'='B',
  'interface_ellipse_eccent_quants_0.9'='B',
  'interface_ellipse_area_quants_0.1'='B',
  'interface_ellipse_area_quants_0.5'='B',
  'interface_ellipse_area_quants_0.9'='B',
  'cooccurrence_mat_0'='I',
  'cooccurrence_mat_1'='I',
  'cooccurrence_mat_2'='I',
  'cooccurrence_mat_3'='I',
  'cooccurrence_mat_4'='I',
  'cooccurrence_mat_5'='I',
  'cooccurrence_mat_6'='I',
  'brain_compactness'='B',
  'brain_convexity'='B',
  'brain_solidity'='I',
  'convex_overlap'='A',
  'filled_overlap_cnt'='I',
  'filled_overlap_min_area'='A',
  'filled_overlap_max_area'='A',
  'filled_overlap_med_area'='A'
)

abi.mat <- c('A', 'B', 'I') %>% sapply(function(vote.f){
  sub.feats <- indiv.features[, names(ABI_groups[ABI_groups == vote.f])]
  # result <- ggpairs(sub.feats, mapping=aes(col=as.factor(micro.scores$new.truth), alpha=.7))
  # print(result)
  prcomp(sub.feats)$x[,'PC1']
}) %>% as.data.frame()

abi.cor <- as.dist(1 - cor(t(abi.mat)))
abi.clust <- hclust(abi.cor)
abi.groups <- cutree(abi.clust, k=3)

ggpairs(abi.mat, mapping=aes(col=as.factor(micro.scores$new.truth), alpha=.7))
ggpairs(abi.mat, mapping=aes(col=as.factor(abi.groups), alpha=.7))
# randomForest(score ~ . - case - part - whole_brain_area - whole_tumor_area, indiv.features %>% mutate(score = as.factor(micro.scores$new.truth)))

ABI.vote.df <- c('A', 'B', 'I') %>% sapply(function(vote.f){
  sub.feats <- feature.groups[, names(ABI_groups[ABI_groups == vote.f])]
  factor.vote <- sub.feats %>% apply(1, function(row){
    which.max(table(factor(row, levels = 1:3)))
  })
  
  factor.vote
}) %>% apply(1, function(row){
  which.max(table(factor(row, levels = 1:3)))
})

ABI.confusion <- confusionMatrix(
  table(
    as.factor(ABI.vote.df),
    as.factor(micro.scores$new.truth)
  ),
  mode='prec_recall'
)

ABI.pca.vote <- pca.res$x %>% as.data.frame() %>%
  mutate(
    case = indiv.features$case,
    score=micro.scores$new.truth,
    estimate=ABI.vote.df,
    interface = sqrt(mapply(min, indiv.features$whole_brain_area, indiv.features$whole_tumor_area))
  )

ggplot(ABI.pca.vote, aes(PC1, PC2, col=as.factor(score))) + geom_point(size=3)
ggplot(ABI.pca.vote, aes(PC1, PC2, col=as.factor(estimate))) + geom_point(size=3)
ggpairs(ABI.pca.vote, columns=1:5, mapping=aes(col=as.factor(micro.scores$new.truth), alpha=.7))

######################################
shifted.feats <- indiv.features[5:ncol(indiv.features)] - min(indiv.features[5:ncol(indiv.features)])
nmf.features <- nmf(shifted.feats, 3, method='snmf/l')
nmf.scores <- nmf.features@fit@W %>% apply(1, which.max)
dists <- 1:3 %>% sapply(function(i){
  diff <- shifted.feats - nmf.features@fit@H[rep(i, nrow(shifted.feats)), ]
  diff %>% apply(1, function(x){norm(x, type='2')})
})

nmf.scores <- dists %>% apply(1, which.min)


nmf.confusion <- confusionMatrix(
  table(
    as.factor(nmf.groups),
    as.factor(micro.scores$new.truth)
  ),
  mode='prec_recall'
)

nmf.cor <- as.dist(1 - cor(t(nmf.features@fit@W)))
nmf.clust <- hclust(nmf.cor)
nmf.groups <- cutree(nmf.clust, k=3)

ggpairs(as.data.frame(nmf.features@fit@W), mapping=aes(col=as.factor(micro.scores$new.truth), alpha=.7))
ggpairs(as.data.frame(nmf.features@fit@W), mapping=aes(col=as.factor(nmf.groups), alpha=.7))
ggplot(pca.vote, aes(PC1, PC2, col=as.factor(score))) + geom_point(size=3)
ggplot(pca.vote, aes(PC1, PC2, col=as.factor(nmf.scores))) + geom_point(size=3)

######################################

selecteds <- c(
  'tumor_components_max_area',
  'brain_solidity',
  'tumor_components_cnt',
  'delaunay_score_quartile3',
  'tumor_components_med_area',
  'brain_compactness',
  'cooccurrence_mat_5',
  'brain_convexity',
  'delaunay_score_median'
)

indiv.features %>% select(all_of(selecteds)) %>%
  mutate(score = micro.scores$new.truth, .before=everything()) %>%
  write.csv('dnn_data.csv', row.names = F)

pca.selected <- indiv.features %>%
  select(all_of(selecteds)) %>% prcomp

selected.vote <- feature.groups %>% select(all_of(selecteds)) %>%
  apply(1, function(row){
    which.max(table(factor(row, levels = 1:3)))
    # round(mean(row))
    # table(factor(row, levels = 1:3)) / length(row)
  })

selected.vote.df <- pca.selected$x %>% as.data.frame() %>%
  mutate(
    case = indiv.features$case,
    score=micro.scores$new.truth,
    estimate=selected.vote,
    interface = sqrt(mapply(min, indiv.features$whole_brain_area, indiv.features$whole_tumor_area))
  )

selected.confusion <- confusionMatrix(
  table(
    as.factor(selected.vote),
    as.factor(micro.scores$new.truth)
  ),
  mode='prec_recall'
)

ggplot(selected.vote.df, aes(PC1, PC2, col=as.factor(score))) + geom_point(size=3)
ggplot(selected.vote.df, aes(PC1, PC2, col=as.factor(estimate))) + geom_point(size=3)


######################################
prob.df <- feature.groups %>% apply(1, function(row){
  table(factor(row, levels = 1:3)) / length(row)
}) %>% t %>% as.data.frame()
colnames(prob.df) <- c('minimal', 'moderate', 'high')

bin.labels <- data.frame(item = 1:length(micro.scores$new.truth), val=1, truth = micro.scores$new.truth) %>%
  dcast(item~truth, fill=0, value.var = 'val') %>% select(-item)


expecteds <- list()
for(patient.id in unique(indiv.features$case)){
  print(paste('Case:', patient.id))
  
  # finding probable combinations
  t.prob.df <- prob.df %>% filter(indiv.features$case == patient.id)
  curr.dist <- unlist(c(t.prob.df[nrow(t.prob.df),], 0))
  curr.label <- matrix(c(1:3, 0))
  
  threshold <- .3
  
  for(i in (nrow(t.prob.df)-1):1){
    last.misc <- curr.dist[length(curr.dist)]
    
    curr.label <- lapply(1:3, function(j){
      cbind(curr.label[-nrow(curr.label),], j)
    })
    curr.label <- Reduce(rbind, curr.label)
    colnames(curr.label) <- NULL
    
    curr.dist <- lapply(1:3, function(j){
      curr.dist[-length(curr.dist)] * t.prob.df[i,j]
    }) %>% unlist
    names(curr.dist) <- NULL
    
    threshold <- quantile(curr.dist, probs = (0:3)/3)[3] * .9
    if(length(curr.dist) > 1e6){
      threshold <- threshold * 1.12
    }
    
    prune <- curr.dist < threshold
    misc <- sum(curr.dist[prune])
    curr.label <- rbind(curr.label[!prune, ], 0)
    curr.dist <- c(curr.dist[!prune], last.misc+misc)
  }
  print(paste0('Left room for ', round(log(length(curr.dist), base=3)), ' out of ', nrow(t.prob.df), ' random guesses'))
  print(paste0('Proceeding with ', length(curr.dist), ' probable scoring'))
  
  ### calculating score
  areas <- indiv.features %>% filter(case == patient.id) %>%
    select(whole_brain_area, whole_tumor_area) %>%
    mutate(interface = sqrt(mapply(min, whole_brain_area, whole_tumor_area))) %>%
    mutate(percent = interface / sum(interface)) %>% select(percent)
    
  curr.dist <- curr.dist[-length(curr.dist)] / sum(curr.dist[-length(curr.dist)])
  curr.label <- curr.label[-nrow(curr.label),]
  possible.scores <- curr.label %>% apply(1, function(scores){
    res <- areas %>% mutate(score = rev(scores)) %>% group_by(score) %>%
      summarise(total_area = sum(percent))
    available.scores <- (1:3)[unique(scores)]
    final.vec <- rep(0, 3)
    final.vec[available.scores] <- res$total_area
    
    final.vec
  }) %>% t
  
  expecteds[[patient.id]] <- possible.scores %>% apply(2, function(curr.score){
    sum(curr.dist * curr.score)
  })
}

prob.dist.decision <- Reduce(rbind.data.frame, expecteds)
colnames(prob.dist.decision) <- c('1', '2', '3')
prob.dist.decision <- prob.dist.decision %>% mutate(case = my.grades$case, .before=everything()) %>%
  mutate(
    domin_s = apply(prob.dist.decision, 1, function(row){which.max(row[-1])}),
    high_s = apply(prob.dist.decision, 1, function(row){max((1:3)[row[-1] > .15])})
  )


plate.conf.prob.d <- confusionMatrix(
  table(
    factor(prob.dist.decision$domin_s, levels=1:3),
    factor(my.grades$domin_s, levels=1:3)
  ),
  mode='prec_recall'
)

plate.conf.prob.h <- confusionMatrix(
  table(
    factor(prob.dist.decision$high_s, levels=1:3),
    factor(my.grades$high_s, levels=1:3)
  ),
  mode='prec_recall'
)




indiv.features %>% group_by(case) %>% mutate(parts=max(part)) %>%
  ungroup() %>% filter(vote.df != micro.scores$new.truth) %>%
  group_by(case) %>% summarise(p=n()/parts) %>% group_by(case) %>%
  summarise(p=mean(p)) %>% View



######################################
for(inquiry in grades$case)
{
  rmarkdown::render(
    'justification.Rmd',
    params=list(patient.id = inquiry),
    output_file = paste0('inquiries/Case', inquiry)#, ' ', Sys.time(), '.html')
  )
}

