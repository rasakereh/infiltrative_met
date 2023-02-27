setwd('/Users/rasakereh/Desktop/uni/infiltrative_met/src')

require(dplyr)
require(ggplot2)
require(limma)
require(stringr)
require(caret)
require(kernlab)
require(infotheo)
require(umap)
require(limma)
require(reshape2)
require(NMF)
require(GGally)
require(pheatmap)
require(boot)

cluster.cnt <- 3

grades <- read.csv('new_grades.csv')

z.normalization <- function(dataset){
  col.n <- colnames(dataset)
  row.n <- rownames(dataset)
  
  z.normed <- dataset %>% t %>% apply(2, scale) %>% t
  colnames(z.normed) <- col.n
  rownames(z.normed) <- row.n
  
  z.normed
}

reduceDim.sd <- function(mat, final.dim=1000)
{
  genes <- apply(mat, 1, function(x){abs(sd(x, na.rm=T))}) %>% sort(decreasing=T) %>% names()
  mat[sort(genes[1:final.dim]),]
}

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


death_date <- read.csv('../data/death_date.csv')
death_date$Date.of.secondary.Dx <- as.Date(death_date$Date.of.secondary.Dx, format = "%m/%d/%Y")
death_date$Date.of.Death...Cerner. <- as.Date(death_date$Date.of.Death...Cerner., format = "%Y-%m-%d")
death_date <- death_date %>% mutate(survival = round(
  as.numeric(difftime(Date.of.Death...Cerner., Date.of.secondary.Dx, units = "days")
  )/30.44)) %>% rename(case = Study..) %>%
  select(case, survival) %>% mutate(cencored = is.na(survival)) %>%
  mutate(case = as.integer(substr(case, 5, 7)))

################################################################################
interface.percent <- indiv.features %>% select(case, part, whole_brain_area, whole_tumor_area) %>%
  mutate(interface = sqrt(mapply(min, whole_brain_area, whole_tumor_area))) %>%
  group_by(case) %>% mutate(percent = interface / sum(interface)) %>% ungroup()

plate.summary <- unique(indiv.features$case) %>% lapply(function(case.num){
  feature.summary <- indiv.features %>% filter(case == case.num) %>%
    select(-case, -part, -whole_brain_area, -whole_tumor_area) %>%
    apply(2, function(col){quantile(col)}) %>% melt() %>%
    mutate(feature = mapply(paste, Var2, Var1), .before=everything()) %>%
    select(-Var2, -Var1) %>% t
  feature.names <- c('case', feature.summary[1,])
  feature.summary <- c(case.num, as.numeric(feature.summary[2,])) %>% matrix(nrow=1) %>% as.data.frame()
  colnames(feature.summary) <- feature.names
  
  feature.summary
})
plate.summary <- Reduce(rbind.data.frame, plate.summary)

###############################################
plate.summary <- plate.summary %>% inner_join(death_date, by='case') %>%
  mutate(available=!cencored) %>% select(-cencored, -case)


plate.cor <- as.dist(1 - cor(plate.summary %>% select(-survival, -available)))
plate.clust <- hclust(plate.cor)
plate.groups <- cutree(plate.clust, k=16)
# plate.PC <- 1:max(plate.groups) %>% sapply(function(clust){
#   sub.feats <- plate.summary[names(plate.groups[plate.groups == clust])] %>% prcomp()
#   sub.feats$x[,'PC1']
# }) %>% as.data.frame() %>%
#   mutate(survival=plate.summary$survival, available=plate.summary$available)

annot <- read.csv("../data/annot.csv")
annot <- annot %>% filter(!is.na(ROI..label.))

annot2smpl <- paste0(annot$Scan.name, '|', str_pad(annot$ROI..label., 3, pad = "0"), '|', gsub(" ", ".", annot$Segment..Name..Label.))
annot2smpl <- cbind.data.frame(annots=annot2smpl, groups=as.factor(gsub("[0-9]*", "", annot$Study.group)), case=annot$Study.Case..)

q3 <- read.csv("../data/Q3.csv", check.names=F)
rownames(q3)<-(q3$TargetName)
q3 <- q3 %>% select(-TargetName)
q3 <- normalizeQuantiles(log(q3))

brain.label <- annot2smpl %>% filter(groups=='LB') %>% select(annots, case) %>%
  inner_join(grades, by='case') %>% arrange(case)
me.label <- annot2smpl %>% filter(groups=='ME') %>% select(annots, case) %>%
  inner_join(grades, by='case') %>% arrange(case)
tilb.label <- annot2smpl %>% filter(groups=='TIL-B') %>% select(annots, case) %>%
  inner_join(grades, by='case') %>% arrange(case)

brain.expr <- reduceDim.sd(q3[brain.label$annots], final.dim=1500)
nmf.brain.expr <- nmf(brain.expr, 10, method = 'snmf/l', seed=1401)
pheatmap(nmf.brain.expr@fit@W)
ggpairs(as.data.frame(nmf.brain.expr@fit@H %>% t), mapping = aes(col=factor(brain.label$domin_s)), columns = 1:10)
ggpairs(as.data.frame(nmf.brain.expr@fit@H %>% t), mapping = aes(col=factor(brain.label$high_s)), columns = 1:10)
nmf.brain.pca <- prcomp(nmf.brain.expr@fit@H %>% t, scale. = T)$x %>% as.data.frame()
ggplot(nmf.brain.pca, aes(PC1, PC2, col=factor(brain.label$domin_s))) + geom_point(size=3)
ggplot(nmf.brain.pca, aes(PC1, PC2, col=factor(brain.label$high_s))) + geom_point(size=3)
nmf.brain.expr@fit@H %>% t %>% as.data.frame() %>%
  mutate(case=brain.label$case, ds=brain.label$domin_s) %>% filter(ds != 3) %>%
  melt(id=c('case', 'ds')) %>%
  ggplot(aes(variable, value, fill=factor(ds))) + geom_boxplot()
nmf.brain.expr@fit@H %>% t %>% as.data.frame() %>%
  mutate(case=brain.label$case, hs=brain.label$high_s) %>%
  melt(id=c('case', 'hs')) %>%
  ggplot(aes(variable, value, fill=factor(hs))) + geom_boxplot()

prcomp(brain.expr %>% t, scale. = T)$x %>% as.data.frame() %>% 
  ggplot(aes(PC1, PC2, col=factor(brain.label$high_s))) + geom_point(size=3)


me.expr <- reduceDim.sd(q3[me.label$annots], final.dim=1500)
nmf.me.expr <- nmf(me.expr, 10, method = 'snmf/l', seed=1401)
pheatmap(nmf.me.expr@fit@W)
ggpairs(as.data.frame(nmf.me.expr@fit@H %>% t), mapping = aes(col=factor(me.label$domin_s)), columns = 1:10)
ggpairs(as.data.frame(nmf.me.expr@fit@H %>% t), mapping = aes(col=factor(me.label$high_s)), columns = 1:10)
nmf.me.pca <- prcomp(nmf.me.expr@fit@H %>% t, scale. = T)$x %>% as.data.frame()
ggplot(nmf.me.pca, aes(PC1, PC2, col=factor(me.label$domin_s))) + geom_point(size=3)
ggplot(nmf.me.pca, aes(PC1, PC2, col=factor(me.label$high_s))) + geom_point(size=3)


tilb.expr <- reduceDim.sd(q3[tilb.label$annots], final.dim=1500)
nmf.tilb.expr <- nmf(tilb.expr, 5, method = 'snmf/l', seed=1401)
pheatmap(nmf.tilb.expr@fit@W)
ggpairs(as.data.frame(nmf.tilb.expr@fit@H %>% t), mapping = aes(col=factor(tilb.label$domin_s)), columns = 1:5)
ggpairs(as.data.frame(nmf.tilb.expr@fit@H %>% t), mapping = aes(col=factor(tilb.label$high_s)), columns = 1:5)
nmf.tilb.pca <- prcomp(nmf.tilb.expr@fit@H %>% t, scale. = T)$x %>% as.data.frame()
ggplot(nmf.tilb.pca, aes(PC1, PC2, col=factor(tilb.label$domin_s))) + geom_point(size=3)
ggplot(nmf.tilb.pca, aes(PC1, PC2, col=factor(tilb.label$high_s))) + geom_point(size=3)


get.relevant.factors2 <- function(nmf.expr, patient.groups){
  single.factor.effect <- nmf.expr %>% apply(2, function(factor.i){
    group.a <- factor.i[patient.groups == levels(patient.groups)[1]]
    group.b <- factor.i[patient.groups == levels(patient.groups)[2]]
    ks.test(group.a, group.b)$p.value #t.test, ks.test, hv.test
  }) %>% sort()
  
  return(single.factor.effect)
}

nmf.brain.expr@fit@H %>% t %>% as.data.frame %>%
  filter(brain.label$domin_s != 3) %>%
  get.relevant.factors2(factor(brain.label$domin_s[brain.label$domin_s != 3]))



get.relevant.factors <- function(nmf.expr, patient.groups, use.all=F, clust.cnt=7, boot.cnt=1000){
  patients.a <- rownames(nmf.expr)[patient.groups == levels(patient.groups)[1]]
  patients.b <- rownames(nmf.expr)[patient.groups == levels(patient.groups)[2]]
  
  if(use.all){
    factor2analyse <- 1:ncol(nmf.expr)
  }else{
    factor.cors <- (1 - (nmf.expr %>% cor() %>% abs)) %>% dist
    factor.clusts <- factor.cors %>% hclust
    # factor.dendograms <- factor.clusts %>% as.dendrogram()
    # possible.colors <- c('red', 'blue', 'orange', 'green', 'purple', 'black', 'pink', 'yellow', 'red', 'blue')
    # factor.dendograms %>% set("branches_k_color", k = clust.cnt, value=possible.colors) %>% plot()
    clust.labels <- factor.clusts %>% cutree(k=clust.cnt)
    factor2analyse <- split(names(clust.labels), clust.labels) %>% sapply(function(cl){cl[1]}) %>% substr(2, 10) %>% as.integer()
  }
  
  factor.tuples <- expand.grid(factor2analyse, factor2analyse, factor2analyse) %>%#, factor2analyse) %>%
    filter(Var1 < Var2 & Var2 < Var3)# & Var3 < Var4)
  factor.scores <- factor.tuples %>% apply(1, function(ftuple){
    a.data <- nmf.expr[patients.a,ftuple] %>% as.matrix()
    b.data <- nmf.expr[patients.b,ftuple] %>% as.matrix()
    
    pca <- prcomp(rbind(a.data, b.data))$x
    ks.test(pca[patients.a,'PC1'], pca[patients.b,'PC1'])$p.value
    #ks2d(pca, fast.patients, slow.patients)
    #wilcox.test(pca[fast.patients,'PC1'], pca[slow.patients,'PC1'])$p.value
    #-abs(median(pca[fast.patients,'PC1']) - median(pca[slow.patients,'PC1']))
    
    # cramer.test(fasts, slows)$p.value
  })
  
  cnt <<- 0
  boot.res <- boot(
    c(patients.a, patients.b),
    function(original, indices){
      g1 <- original[indices[1:length(patients.a)]]
      g2 <- original[indices[(length(patients.a)+1):nrow(nmf.expr)]]
      cnt <<- cnt+1
      if(cnt %% (boot.cnt %/% 100 * 5) == 0){
        write(paste0('boot: ', round(cnt/boot.cnt * 100), '%'), stderr())
      }
      
      factor.scores <- factor.tuples %>% apply(1, function(ftuple){
        a.data <- nmf.expr[g1,ftuple] %>% as.matrix()
        b.data <- nmf.expr[g2,ftuple] %>% as.matrix()
        
        pca <- prcomp(rbind(a.data, b.data))$x
        ks.test(pca[g1,'PC1'], pca[g2,'PC1'])$p.value
        #wilcox.test(pca[g1,'PC1'], pca[g2,'PC1'])$p.value
        #ks2d(pca, g1, g2)
        #-abs(median(pca[g1,'PC1']) - median(pca[g2,'PC1']))
        
        # cramer.test(fasts, slows)$p.value
      }) %>% min()
    },
    boot.cnt
  )
  all.res <- boot.res$t
  p.val <- (sum(min(factor.scores) > all.res)+1)/(length(all.res)+1)
  
  all.tuples <- factor.tuples[min(factor.scores) == factor.scores,]
  
  separator.factors.int <- all.tuples[1,] %>% unlist()
  separator.factors <- paste0('Factor', separator.factors.int)
  
  list(char=separator.factors, int=separator.factors.int, p.values=p.val, all.possibles=all.tuples)
}

ds.relevant.factors <- nmf.brain.expr@fit@H %>% t %>% as.data.frame %>%
  filter(brain.label$domin_s != 3) %>%
  get.relevant.factors(factor(brain.label$domin_s[brain.label$domin_s != 3]), use.all=T, boot.cnt=2000)
ds.factors <- c(2,5,9) #old: (2, 5, 3, 10)
ggplot(
  prcomp(nmf.brain.expr@fit@H[ds.factors,] %>% t, scale. = T)$x %>% as.data.frame(),
  aes(PC1, PC2, col=factor(brain.label$domin_s))
) + geom_point(size=3)
pheatmap(nmf.brain.expr@fit@W[,ds.factors])
nmf.brain.expr@fit@W[,ds.factors] %>% apply(2, sort) %>% as.data.frame() %>%
  mutate(index=1:nrow(nmf.brain.expr@fit@W)) %>% melt(id='index') %>%
  ggplot(aes(index, value)) + geom_point() + facet_wrap(~variable)
nmf.brain.expr@fit@W[,ds.factors] %>% apply(2, function(values){
  rownames(nmf.brain.expr@fit@W)[order(values, decreasing = T)][1:100]
}) %>% write.csv('ds_genes.csv')
ggpairs(
  nmf.brain.expr@fit@H[ds.factors,] %>% t %>% as.data.frame %>% filter(brain.label$domin_s != 3),
  mapping = aes(col=factor(brain.label$domin_s[brain.label$domin_s != 3]))
) #high ~ bad: 2, 5; low ~ bad: 9


#old: (1, 4, 5, 10)
hs.relevant.factors <- nmf.brain.expr@fit@H %>% t %>% as.data.frame %>%
  filter(brain.label$high_s != 1) %>%
  get.relevant.factors(factor(brain.label$high_s[brain.label$high_s != 1]), use.all=T, boot.cnt=2000)
hs.factors <- c(4,5,10)
ggplot(
  prcomp(nmf.brain.expr@fit@H[hs.factors,] %>% t, scale. = T)$x %>% as.data.frame(),
  aes(PC1, PC2, col=factor(brain.label$high_s))
) + geom_point(size=3)
pheatmap(nmf.brain.expr@fit@W[,hs.factors])
nmf.brain.expr@fit@W[,hs.factors] %>% apply(2, sort) %>% as.data.frame() %>%
  mutate(index=1:nrow(nmf.brain.expr@fit@W)) %>% melt(id='index') %>%
  ggplot(aes(index, value)) + geom_point() + facet_wrap(~variable)
nmf.brain.expr@fit@W[,hs.factors] %>% apply(2, function(values){
  rownames(nmf.brain.expr@fit@W)[order(values, decreasing = T)][1:100]
}) %>% write.csv('hs_genes.csv')
ggpairs(
  nmf.brain.expr@fit@H[hs.factors,] %>% t %>% as.data.frame %>% filter(brain.label$high_s != 1),
  mapping = aes(col=factor(brain.label$high_s[brain.label$high_s != 1]))
) #high ~ bad: 5; low ~ bad: 4, 10

