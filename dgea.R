setwd('/Users/rasakereh/Desktop/uni/infiltrative_met/src')

require(dplyr)
require(ggplot2)
require(limma)
require(stringr)
require(caret)
require(kernlab)
require(infotheo)
require(umap)

cluster.cnt <- 3

features <- read.csv('reshaped_dual.csv')
grades <- features %>% select(case, domin_s, d_per, high_s, h_per, total)
features <- features %>% select(-domin_s, -d_per, -high_s, -h_per, -total)
no.var.cols <- colnames(features)[features %>% apply(2, sd) == 0]
features <- features %>% select(-all_of(no.var.cols))
# features <- features %>% select(matches('(case|nearest|surroundedness|overlap)'))
pca_res <- prcomp(features %>% select(-case), scale=T)$x
pca_feat <- data.frame(
  case = features$case,
  pca_res,
  invasive = cut(grades$total, quantile(grades$total, (0:cluster.cnt)/cluster.cnt), include.lowest=T) %>% as.integer %>% as.factor()#ifelse(grades$domin_s == 1, 'minimal', ifelse(grades$domin_s == 2, 'moderate', 'high')) %>% as.factor()
)


# specc(pca_feat %>% select(-case, -invasive), centers=2)

mutinformation(pca_feat %>% select(PC1, PC2, PC3) %>% discretize, discretize(pca_feat$invasive))
mutinformation(pca_feat %>% select(-case, -invasive) %>% discretize, discretize(pca_feat$invasive))

patient.cor <- as.dist(1 - cor(t(pca_feat %>% select(-case, -invasive))))
patient.cor <- as.dist(1 - cor(t(pca_feat %>% select(PC1, PC2, PC3))))
pc.based.clust <- hclust(patient.cor)
plot(pc.based.clust)
patient.groups <- cutree(pc.based.clust, k=cluster.cnt)

pca_feat <- cbind.data.frame(pca_feat, estimated=as.factor(patient.groups))
umap_feat <- umap(features %>% select(-case))$layout %>% as.data.frame()
umap_feat <- cbind.data.frame(umap_feat, invasive=pca_feat$invasive)

# ggplot(pca_feat, aes(PC1, PC2, col=invasive, shape=estimated)) + geom_point(size=5) + scale_shape_manual(values=c(0, 1, 2, 5))
ggplot(pca_feat, aes(PC1, PC2, col=invasive)) + geom_point(size=5)
ggplot(umap_feat, aes(V1, V2, col=invasive)) + geom_point(size=5)
conf.mat <- confusionMatrix(
  table(
    as.factor(as.integer(pca_feat$estimated)),
    as.factor(as.integer(pca_feat$invasive))
  ),
  positive = '4',
  mode='prec_recall'
)


###############################
# load series and platform data from GEO
annot <- read.csv("../data/annot.csv")
annot <- annot %>% filter(!is.na(ROI..label.))

annot2smpl <- paste0(annot$Scan.name, '|', str_pad(annot$ROI..label., 3, pad = "0"), '|', gsub(" ", ".", annot$Segment..Name..Label.))
annot2smpl <- cbind.data.frame(annots=annot2smpl, groups=as.factor(gsub("[0-9]*", "", annot$Study.group)), case=annot$Study.Case..)

q3 <- read.csv("../data/Q3.csv", check.names=F)
rownames(q3)<-(q3$TargetName)
q3 <- q3 %>% select(-TargetName)
q3 <- log(q3)
pca.dim <- 5
pca <- prcomp(t(q3))
pca.rotation <- pca$r
pca <- cbind.data.frame(pca$x[,1:pca.dim], annots=rownames(pca$x))
pca <- inner_join(pca, annot2smpl) %>% select(-annots) %>% filter(groups %in% c('LB', 'ME', 'TIL-B'))
pca <- inner_join(pca, pca_feat %>% select(case, invasive, estimated), c(case='case'))

lb.pca <- pca %>% filter(groups == 'LB')
ggplot(lb.pca, aes(PC1, PC2, col=invasive, shape=estimated)) + geom_point(size=5)

do.DGEA <- function(invasive.data, invasive.labels, invasive.annots){
  invasive.groups <- rep("g2", length(invasive.labels))
  names(invasive.groups) <- invasive.annots
  invasive.groups[as.integer(invasive.labels) == 1] <- "g1"
  names(invasive.groups) <- NULL
  
  # assign samples to groups and set up design matrix
  gs <- factor(invasive.groups)
  design <- model.matrix(~gs + 0, invasive.data)
  groups <- levels(gs)
  colnames(design) <- groups
  
  fit <- lmFit(invasive.data, design)  # fit linear model
  
  # set up contrasts of interest and recalculate model coefficients
  cts <- paste(groups[1], groups[2], sep="-")
  cont.matrix <- makeContrasts(contrasts=cts, levels=design)
  fit2 <- contrasts.fit(fit, cont.matrix)
  
  # compute statistics and table of top significant genes
  fit2 <- eBayes(fit2, 0.01)
  tT <- topTable(fit2, adjust="fdr", sort.by="B", number=100)
  
  tT
}

invasive.labels <- annot2smpl %>%
  inner_join(pca_feat, c(case = "case")) %>%
  filter(groups=='ME') %>% select(annots, invasive, estimated) %>%
  mutate(invasive = as.factor(invasive))
invasive.data <- (q3[invasive.labels$annots])
colnames(invasive.data) <- make.names(colnames(invasive.data))

tT.ME.real <- do.DGEA(invasive.data, invasive.labels$invasive, invasive.labels$annots)
tT.ME.estimated <- do.DGEA(invasive.data, invasive.labels$estimated, invasive.labels$annots)


invasive.labels <- annot2smpl %>%
  inner_join(pca_feat, c(case = "case")) %>%
  filter(groups=='LB') %>% select(annots, invasive, estimated) %>%
  mutate(invasive = as.factor(invasive))
invasive.data <- (q3[invasive.labels$annots])
colnames(invasive.data) <- make.names(colnames(invasive.data))

tT.LB.real <- do.DGEA(invasive.data, invasive.labels$invasive, invasive.labels$annots)
tT.LB.estimated <- do.DGEA(invasive.data, invasive.labels$estimated, invasive.labels$annots)

intersect(rownames(tT.ME.real), rownames(tT.ME.estimated))
intersect(rownames(tT.LB.real), rownames(tT.LB.estimated))
