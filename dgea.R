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
require(cramer)
require(circlize)

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
  inner_join(grades, by='case') %>% group_by(case) %>%
  summarise_all(min) %>% arrange(case)
me.label <- annot2smpl %>% filter(groups=='ME') %>% select(annots, case) %>%
  inner_join(grades, by='case') %>% group_by(case) %>%
  summarise_all(min) %>% arrange(case)
tilb.label <- annot2smpl %>% filter(groups=='TIL-B') %>% select(annots, case) %>%
  inner_join(grades, by='case') %>% group_by(case) %>%
  summarise_all(min) %>% arrange(case)

brain.expr <- reduceDim.sd(q3[brain.label$annots], final.dim=1500)
nmf.brain.expr <- nmf(brain.expr, 10, method = 'snmf/l', seed=1401)
pheatmap(nmf.brain.expr@fit@W, show_rownames = F)
ggpairs(as.data.frame(nmf.brain.expr@fit@H %>% t), mapping = aes(col=factor(brain.label$domin_s)), columns = 1:10) + ggtitle("Dominant Scores")
ggpairs(as.data.frame(nmf.brain.expr@fit@H %>% t), mapping = aes(col=factor(brain.label$high_s)), columns = 1:10) + ggtitle("High Scores")
# nmf.brain.pca <- prcomp(nmf.brain.expr@fit@H %>% t, scale. = T)$x %>% as.data.frame()
# ggplot(nmf.brain.pca, aes(PC1, PC2, col=factor(brain.label$domin_s))) +
#   geom_point(size=3) + labs(col = "Dominant Score")
# ggplot(nmf.brain.pca, aes(PC1, PC2, col=factor(brain.label$high_s))) +
#   geom_point(size=3) + labs(col = "High Score")
nmf.brain.expr@fit@H %>% t %>% as.data.frame() %>%
  mutate(case=brain.label$case, ds=brain.label$domin_s) %>% filter(ds != 3) %>%
  melt(id=c('case', 'ds')) %>%
  ggplot(aes(variable, value, fill=factor(ds))) + geom_boxplot() +
  labs(fill = "Dominant Score")
nmf.brain.expr@fit@H %>% t %>% as.data.frame() %>%
  mutate(case=brain.label$case, hs=brain.label$high_s) %>%
  melt(id=c('case', 'hs')) %>%
  ggplot(aes(variable, value, fill=factor(hs))) + geom_boxplot() +
  labs(fill = "High Score")

prcomp(brain.expr %>% t, scale. = T)$x %>% as.data.frame() %>% 
  ggplot(aes(PC1, PC2, col=factor(brain.label$domin_s))) + geom_point(size=3)
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
  
  factor.tuples <- expand.grid(factor2analyse, factor2analyse) %>%#, factor2analyse) %>%#, factor2analyse) %>%
    filter(Var1 < Var2)# & Var2 < Var3)# & Var3 < Var4)
  factor.scores <- factor.tuples %>% apply(1, function(ftuple){
    a.data <- nmf.expr[patients.a,ftuple] %>% as.matrix()
    b.data <- nmf.expr[patients.b,ftuple] %>% as.matrix()
    
    pca <- prcomp(rbind(a.data, b.data))
    ks.test(pca$x[patients.a,'PC1'], pca$x[patients.b,'PC1'])$p.value
    # ws <- pca$sd[1:2] / sum(pca$sd[1:2])
    # ws[1] * ks.test(pca$x[patients.a,'PC1'], pca$x[patients.b,'PC1'])$p.value +
    #   ws[2] * ks.test(pca$x[patients.a,'PC2'], pca$x[patients.b,'PC2'])$p.value
    # wilcox.test(pca[patients.a,'PC1'], pca[patients.b,'PC1'])$p.value
    # HotellingsT2Test(a.data, b.data)$p.value
    #-abs(median(pca[fast.patients,'PC1']) - median(pca[slow.patients,'PC1']))
    
    # cramer.test(a.data, b.data)$p.value
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
        
        pca <- prcomp(rbind(a.data, b.data))
        ks.test(pca$x[g1,'PC1'], pca$x[g2,'PC1'])$p.value
        # ws <- pca$sd[1:2] / sum(pca$sd[1:2])
        # ws[1] * ks.test(pca$x[g1,'PC1'], pca$x[g2,'PC1'])$p.value +
        #   ws[2] * ks.test(pca$x[g1,'PC2'], pca$x[g2,'PC2'])$p.value
        # HotellingsT2Test(a.data, b.data)$p.value
        # wilcox.test(pca[g1,'PC1'], pca[g2,'PC1'])$p.value
        #-abs(median(pca[g1,'PC1']) - median(pca[g2,'PC1']))
        
        # cramer.test(a.data, b.data)$p.value
      }) %>% min()
    },
    boot.cnt
  )
  all.res <- boot.res$t
  nominal.p.val <- min(factor.scores)
  p.val <- (sum(nominal.p.val > all.res)+1)/(length(all.res)+1)
  
  print(ggplot(data.frame(statistic=log10(all.res)), aes(x=all.res)) +
          geom_density() + xlab('statistic') +
          geom_vline(xintercept = log10(nominal.p.val), colour="red", linetype = "longdash"))
  
  all.tuples <- factor.tuples[nominal.p.val == factor.scores,]
  
  separator.factors.int <- all.tuples[1,] %>% unlist()
  separator.factors <- paste0('Factor', separator.factors.int)
  
  list(
    char=separator.factors,
    int=separator.factors.int,
    nominal.p.val=nominal.p.val,
    p.values=p.val,
    all.possibles=all.tuples
  )
}

get.max.genes <- function(weight.mat, factor.tuple){
  sorted.weights <- weight.mat[,factor.tuple] %>% apply(2, sort) %>% as.data.frame()
  colnames(sorted.weights) <- paste('Factor', factor.tuple)
  sorted.weights$position <- 1:nrow(weight.mat)
  sorted.weights <- melt(sorted.weights, id='position')
  threshold.plot <- ggplot(sorted.weights, aes(position, value)) + geom_point() +
    geom_smooth(method='lm', formula=y~x) + facet_wrap(~variable, ncol=1)
  print(threshold.plot)
  gene.sets <- sapply(factor.tuple, function(curr.factor){
    curr.weight <- weight.mat[,curr.factor] %>% sort()
    cwl <- length(curr.weight)
    cwdf <- data.frame(x=1:cwl, y=curr.weight)
    cwmdl <- lm(y ~ x, data = cwdf)
    linear.cw <- predict(cwmdl, cwdf[1])
    lin.resid <- curr.weight - linear.cw
    
    total.cnt <- cwl - max(which(lin.resid <= 0))
    rownames(weight.mat)[order(weight.mat[,curr.factor], decreasing = T)][1:total.cnt]
  })
  
  gene.sets <- sapply(gene.sets, function(vec){
    length(vec) <- max(sapply(gene.sets, length))
    dim(vec) <- c(length(vec), 1)
    vec
  })
  gene.sets[is.na(gene.sets)] <- ''
  
  gene.sets
}

ds.relevant.factors <- nmf.brain.expr@fit@H %>% t %>% as.data.frame %>%
  filter(brain.label$domin_s != 3) %>%
  get.relevant.factors(factor(brain.label$domin_s[brain.label$domin_s != 3]), use.all=T, boot.cnt=1500)
ds.factors <- c(1, 3, 9) #c(1, 3, 9) #old: (2, 5, 3, 10)
ggplot(
  prcomp(nmf.brain.expr@fit@H[ds.factors,] %>% t, scale. = T)$x %>% as.data.frame(),
  aes(PC1, PC2, col=factor(brain.label$domin_s))
) + geom_point(size=3) + labs(col='Dominant Score')
pheatmap(nmf.brain.expr@fit@W[,ds.factors])
nmf.brain.expr@fit@W[,ds.factors] %>% apply(2, sort) %>% as.data.frame() %>%
  setNames(paste0('F', ds.factors)) %>%
  mutate(index=1:nrow(nmf.brain.expr@fit@W)) %>% melt(id='index') %>%
  ggplot(aes(index, value)) + geom_point() + facet_wrap(~variable) +
  ggtitle("Dominant Scores")

ggpairs(
  nmf.brain.expr@fit@H[ds.factors,] %>% t %>% as.data.frame %>% filter(brain.label$domin_s != 3),
  mapping = aes(col=factor(brain.label$domin_s[brain.label$domin_s != 3]))
) #high ~ bad: 2, 5; low ~ bad: 9


#old: (1, 4, 5, 10)
hs.relevant.factors <- nmf.brain.expr@fit@H %>% t %>% as.data.frame %>%
  filter(brain.label$high_s != 1) %>%
  get.relevant.factors(factor(brain.label$high_s[brain.label$high_s != 1]), use.all=T, boot.cnt=1500)
hs.factors <- c(4, 6, 10) #c(4,6,10)
ggplot(
  prcomp(nmf.brain.expr@fit@H[hs.factors,] %>% t, scale. = T)$x %>% as.data.frame(),
  aes(PC1, PC2, col=factor(brain.label$high_s))
) + geom_point(size=3) + labs(col='High Score')
pheatmap(nmf.brain.expr@fit@W[,hs.factors])
nmf.brain.expr@fit@W[,hs.factors] %>% apply(2, sort) %>% as.data.frame() %>%
  setNames(paste0('F', hs.factors)) %>%
  mutate(index=1:nrow(nmf.brain.expr@fit@W)) %>% melt(id='index') %>%
  ggplot(aes(index, value)) + geom_point() + facet_wrap(~variable) +
  ggtitle("High Scores")

ggpairs(
  nmf.brain.expr@fit@H[hs.factors,] %>% t %>% as.data.frame %>% filter(brain.label$high_s != 1),
  mapping = aes(col=factor(brain.label$high_s[brain.label$high_s != 1]))
) #high ~ bad: 5; low ~ bad: 4, 10

########## new

ds.signif.genes <- get.max.genes(nmf.brain.expr@fit@W, ds.factors)
ds.signif.genes %>% write.csv('ds_genes.csv')

all.ds.genes <- unique(c(ds.signif.genes))
all.ds.genes <- all.ds.genes[all.ds.genes != '']

hs.signif.genes <- get.max.genes(nmf.brain.expr@fit@W, hs.factors)
hs.signif.genes %>% write.csv('hs_genes.csv')
all.hs.genes <- unique(c(hs.signif.genes))
all.hs.genes <- all.hs.genes[all.hs.genes != '']


ds.expr <- as.data.frame(
  nmf.brain.expr@fit@W[,ds.factors] %*% nmf.brain.expr@fit@H[ds.factors,]
)

patient_heatmap <- pheatmap(
  ds.expr[all.ds.genes,],
  show_rownames = F, labels_col = brain.label$domin_s,
  border_color = NA, main='Dominant Scores'
)
pclusts.ds <- cutree(patient_heatmap$tree_col, k = 3)
pclusts.ds[pclusts.ds == 3] <- 1

clust.grade.ds <- brain.label$domin_s
clust.grade.ds[clust.grade.ds == 3] <- 2

fisher.test(table(clust.grade.ds, pclusts.ds))


hs.expr <- as.data.frame(
  nmf.brain.expr@fit@W[,hs.factors] %*% nmf.brain.expr@fit@H[hs.factors,]
)

patient_heatmap <- pheatmap(
  hs.expr[all.hs.genes,],
  show_rownames = F, labels_col = brain.label$high_s,
  border_color = NA, main='High Scores'
)
pclusts.hs <- cutree(patient_heatmap$tree_col, k = 3)
pclusts.hs[pclusts.hs == 3] <- 2
pclusts.hs[pclusts.hs == 1] <- 3

clust.grade.hs <- brain.label$high_s

fisher.test(table(clust.grade.hs, pclusts.hs))


# pheatmap(
#   brain.expr[c(hs.signif.genes), pclusts.hs!=3],
#   show_rownames = F, labels_col = paste(brain.label$high_s[pclusts != 3], paste0('gp', pclusts[pclusts != 3]), sep=':')
# )



circos.plot <- function(gene.list, relevant.factors, plot.title){
  colnames(gene.list) <- paste0('Factor', relevant.factors)
  gene.list <- melt(gene.list) %>% rename(Rank=Var1, Factor=Var2, Gene=value) %>%
    filter(Gene != '') %>%
    mutate(unique.g = sprintf('%s (%s)', Gene, substring(Factor, nchar('Factor')+1)))
  
  correlations <- brain.expr[gene.list$Gene, ] # ds.expr[gene.list$Gene, ]
  correlations <- cor(t(correlations))
  heatmap(correlations)
  correlations <- melt(correlations) %>% filter(abs(value) > .7, as.character(Var1) < as.character(Var2))
  correlations <- correlations %>% inner_join(gene.list, c('Var1'='Gene')) %>%
    inner_join(gene.list, c('Var2'='Gene')) %>% select(-Rank.x, -Rank.y) %>%
    mutate(Factor.x=as.character(Factor.x), Factor.y=as.character(Factor.y)) %>%
    filter(Factor.x != Factor.y)
  
  pivot.genes <- sort(table(c(correlations$Var1, correlations$Var2)), decreasing=T)
  pivot.genes <- pivot.genes[pivot.genes > 2]
  
  correlations <- correlations %>%
    filter(Var1 %in% names(pivot.genes) | Var2 %in% names(pivot.genes))
  
  pivot.genes <- pivot.genes[pivot.genes > 10]
  
  
  factor.list = c(structure(correlations$Factor.x, names=correlations$unique.g.x),
                  structure(correlations$Factor.y,names= correlations$unique.g.y))
  factor.list = factor.list[!duplicated(names(factor.list))]
  factor.list = factor.list[order(factor.list, names(factor.list))]
  factor.list_color = structure(seq(2,length(unique(factor.list))+1), names = unique(factor.list))
  genes_color = structure(seq(2,length(names(factor.list))+1), names = names(factor.list))
  
  gap.degree = do.call("c", lapply(table(factor.list), function(i) c(rep(2, i-1), 8)))
  circos.par(gap.degree = gap.degree)
  
  chordDiagram(correlations[, c('unique.g.x', 'unique.g.y')], order = names(factor.list), grid.col = genes_color,
               directional = 1, annotationTrack = "grid", preAllocateTracks = list(
                 list(track.height = 0.02))
  )
  
  
  circos.trackPlotRegion(track.index = 2, panel.fun = function(x, y) {
    xlim = get.cell.meta.data("xlim")
    ylim = get.cell.meta.data("ylim")
    sector.index = get.cell.meta.data("sector.index")
    gene.name = strsplit(sector.index, ' ')[[1]][1]
    if(gene.name %in% names(pivot.genes)){
      circos.text(mean(xlim), mean(ylim), gene.name, col = "white", cex = 0.4, facing = "inside", niceFacing = TRUE)
    }else{
      ''
    }
  }, bg.border = NA)
  
  
  for(b in unique(factor.list)) {
    curr.factor = names(factor.list[factor.list == b])
    highlight.sector(sector.index = curr.factor, track.index = 1, col = factor.list_color[b], 
                     text = b, text.vjust = -1, niceFacing = TRUE)
  }
  title(plot.title, adj = 0.01, line = -1)
  
  circos.clear()
  
  pivot.genes
}

ds.pivot.genes <- circos.plot(ds.signif.genes[1:150, ], ds.factors, 'Dominant Scores')
hs.pivot.genes <- circos.plot(hs.signif.genes[1:150, ], hs.factors, 'High Scores')

pheatmap(brain.expr[names(ds.pivot.genes),], labels_col = brain.label$domin_s, border_color = NA)
pheatmap(brain.expr[names(hs.pivot.genes),], labels_col = brain.label$high_s, border_color = NA)

# V(data) = E[V(data | label)] + V(E[data | label])
X_std <- scale(t(nmf.brain.expr@fit@H)); V_data <- var(X_std)
X_std <- scale(t(nmf.brain.expr@fit@H[,brain.label$domin_s==1])); V_data_1 <- var(X_std)
X_std <- scale(t(nmf.brain.expr@fit@H[,brain.label$domin_s==2])); V_data_2 <- var(X_std)
X_std <- scale(t(nmf.brain.expr@fit@H[,brain.label$domin_s==3])); V_data_3 <- var(X_std)
label_portions <- table(factor(brain.label$domin_s, levels=1:3)) / length(brain.label$domin_s)
V_relevant <- V_data - label_portions[1]*V_data_1 - label_portions[2]*V_data_2 - label_portions[3]*V_data_3
desired_percent <- sum(abs(V_relevant)) / sum(abs(V_data))

X_std <- scale(t(nmf.brain.expr@fit@H[,brain.label$high_s==1])); V_data_1 <- var(X_std); V_data_1[is.na(V_data_1)] <- 0
X_std <- scale(t(nmf.brain.expr@fit@H[,brain.label$high_s==2])); V_data_2 <- var(X_std)
X_std <- scale(t(nmf.brain.expr@fit@H[,brain.label$high_s==3])); V_data_3 <- var(X_std)
label_portions <- table(factor(brain.label$high_s, levels=1:3)) / length(brain.label$high_s)
V_relevant <- V_data - label_portions[1]*V_data_1 - label_portions[2]*V_data_2 - label_portions[3]*V_data_3
desired_percent <- sum(abs(V_relevant)) / sum(abs(V_data))






fast.cut <- 12
slow.cut <- 35

all.factors <- readRDS('Interval__Mths_-data.RDS')
time.labels <- ifelse(all.factors$time <= fast.cut, 1, ifelse(all.factors$time >= slow.cut, 3, 2))
all.factors <- all.factors %>% select(-time, -delta)


X_std <- scale(all.factors); V_data <- var(X_std)
X_std <- scale(all.factors[time.labels==1,]); V_data_1 <- var(X_std)
X_std <- scale(all.factors[time.labels==2,]); V_data_2 <- var(X_std)
X_std <- scale(all.factors[time.labels==3,]); V_data_3 <- var(X_std)
label_portions <- table(factor(time.labels, levels=1:3)) / length(time.labels)
V_relevant <- V_data - label_portions[1]*V_data_1 - label_portions[2]*V_data_2 - label_portions[3]*V_data_3
desired_percent <- sum(abs(V_relevant)) / sum(abs(V_data))

prcomp(all.factors)$x %>% as.data.frame() %>% ggplot(aes(PC1, PC2, col=factor(time.labels))) + geom_point(size=3)





optimum.cmp.cnt <- nmf(brain.expr, seq(8, 18, 2), method = 'snmf/l', seed=1401)




1















survival_times <- read.csv('../data/death_date.csv')
survival_times$Date.of.primary.Dx <- as.Date(survival_times$Date.of.primary.Dx, format = "%m/%d/%Y")
survival_times$Date.of.secondary.Dx <- as.Date(survival_times$Date.of.secondary.Dx, format = "%m/%d/%Y")
survival_times$Date.of.Death...Cerner. <- as.Date(survival_times$Date.of.Death...Cerner., format = "%Y-%m-%d")
survival_times <- survival_times %>% mutate(survival = round(
  as.numeric(difftime(Date.of.Death...Cerner., Date.of.secondary.Dx, units = "days")
  )/30.44), met_time = round(
    as.numeric(difftime(Date.of.secondary.Dx, Date.of.primary.Dx, units = "days")
    )/30.44)) %>% rename(case = Study..) %>%
  select(case, survival, met_time) %>%
  mutate(case = as.integer(substr(case, 5, 7)))
survival_times <- survival_times %>% inner_join(brain.label, by=c('case'='case'))


met_groups <- survival_times$met_time[pclusts != 3]
met_groups <- as.numeric(met_groups < mean(met_groups))+1
surv_groups <- survival_times$survival[pclusts != 3]
surv_groups <- as.numeric(surv_groups < mean(surv_groups, na.rm=T))+1

fisher.test(table(met_groups, pclusts[pclusts != 3]))
fisher.test(table(surv_groups, pclusts[pclusts != 3]))

ggplot(
  data.frame(class=factor(pclusts), score=brain.label$high_s + brain.label$h_per),
  aes(class, score, col=factor(brain.label$high_s))
  ) + geom_boxplot()


ggplot(
  prcomp(nmf.brain.expr@fit@H[hs.factors,] %>% t, scale. = T)$x %>% as.data.frame(),
  aes(PC1, PC2, col=factor(pclusts))
) + geom_point(size=3)




# assign samples to groups and set up design matrix
gset <- brain.expr[, pclusts != 3]
gs <- factor(pclusts[pclusts != 3])
groups <- make.names(c("sg1","sg2"))
levels(gs) <- groups
design <- model.matrix(~gs + 0, NULL)
colnames(design) <- levels(gs)

fit <- lmFit(gset, design)  # fit linear model

# set up contrasts of interest and recalculate model coefficients
cts <- paste(groups[1], groups[2], sep="-")
cont.matrix <- makeContrasts(contrasts=cts, levels=design)
fit2 <- contrasts.fit(fit, cont.matrix)

# compute statistics and table of top significant genes
fit2 <- eBayes(fit2, 0.01)
tT <- topTable(fit2, adjust="fdr", sort.by="B", number=250) %>% filter(adj.P.Val < .05)

cbind(hs.signif.genes, ds.signif.genes) %>% apply(2, function(curr.genes){
  intersect(rownames(tT), curr.genes) %>% length
})

write(rownames(tT), 'deg_subg.txt')

