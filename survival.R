setwd('/Users/rasakereh/Desktop/uni/infiltrative_met/src')

require(dplyr)
require(ggplot2)
require(stringr)
require(infotheo)
require(GGally)
require(survival)
require(randomForestSRC)
require(limma)
require(reshape2)


z.normalization <- function(dataset){
  col.n <- colnames(dataset)
  row.n <- rownames(dataset)
  
  z.normed <- dataset %>% t %>% apply(2, scale) %>% t
  colnames(z.normed) <- col.n
  rownames(z.normed) <- row.n
  
  z.normed
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

death_date <- read.csv('../data/death_date.csv')
death_date$Date.of.primary.Dx <- as.Date(death_date$Date.of.primary.Dx, format = "%m/%d/%Y")
death_date$Date.of.secondary.Dx <- as.Date(death_date$Date.of.secondary.Dx, format = "%m/%d/%Y")
death_date$Date.of.Death...Cerner. <- as.Date(death_date$Date.of.Death...Cerner., format = "%Y-%m-%d")
death_date <- death_date %>% mutate(
  survival = round(as.numeric(difftime(Date.of.Death...Cerner., Date.of.secondary.Dx, units = "days"))/30.44)) %>%
  mutate(
    met_interval = round(as.numeric(difftime(Date.of.secondary.Dx, Date.of.primary.Dx, units = "days"))/30.44)) %>%
  rename(case = Study..) %>%
  select(case, survival, met_interval) %>% mutate(cencored = is.na(survival)) %>%
  mutate(case = as.integer(substr(case, 5, 7)))

################################################################################
interface.percent <- indiv.features %>% select(case, part, whole_brain_area, whole_tumor_area) %>%
  mutate(interface = sqrt(mapply(min, whole_brain_area, whole_tumor_area))) %>%
  group_by(case) %>% mutate(percent = interface / sum(interface)) %>% ungroup()
plate.summary <- indiv.features %>%
  select(-case, -part, -whole_brain_area, -whole_tumor_area) %>%
  mutate_all(function(x){interface.percent$percent * x}) %>%
  group_by(interface.percent$case) %>% summarise_all(sum)
colnames(plate.summary)[1] <- 'case'

plate.summary2 <- unique(indiv.features$case) %>% lapply(function(case.num){
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
plate.summary2 <- Reduce(rbind.data.frame, plate.summary2)

###############################################
plate.summary <- plate.summary %>% inner_join(death_date, by='case') %>%
  mutate(available=!cencored) %>% select(-cencored, -case, -met_interval)
plate.summary2 <- plate.summary2 %>% inner_join(death_date, by='case') %>%
  mutate(available=!cencored) %>% select(-cencored, -case, -met_interval)


cor(plate.summary, plate.summary$survival, use='complete.obs') %>% abs %>% View
ggplot(plate.summary, aes(survival, filled_overlap_max_area)) +
  geom_point(size=3) + geom_smooth(method='lm') + ylim(c(-2, 2))

cor(plate.summary2, plate.summary2$survival, use='complete.obs') %>% abs %>% View
ggplot(plate.summary2, aes(survival, `cooccurrence_mat_3 100%`)) +
  geom_point(size=3) + geom_smooth(method='lm') + ylim(c(-2, 2))
ggplot(plate.summary2, aes(survival, `cooccurrence_mat_3 0%`)) +
  geom_point(size=3) + geom_smooth(method='lm') + ylim(c(-2, 2))
ggplot(plate.summary2, aes(survival, `filled_overlap_max_area 75%`)) +
  geom_point(size=3) + geom_smooth(method='lm') + ylim(c(-2, 2))
ggplot(plate.summary2, aes(survival, `filled_overlap_max_area 100%`)) +
  geom_point(size=3) + geom_smooth(method='lm') + ylim(c(-2, 2))
ggplot(plate.summary2, aes(survival, `cooccurrence_mat_1 75%`)) +
  geom_point(size=3) + geom_smooth(method='lm') + ylim(c(-2, 2))


plate.cor <- as.dist(1 - cor(plate.summary %>% select(-survival, -available)))
plate.clust <- hclust(plate.cor)
plate.groups <- cutree(plate.clust, k=16)
plate.PC <- 1:max(plate.groups) %>% sapply(function(clust){
  sub.feats <- plate.summary[names(plate.groups[plate.groups == clust])] %>% prcomp()
  sub.feats$x[,'PC1']
}) %>% as.data.frame() %>%
  mutate(survival=plate.summary$survival, available=plate.summary$available)

plate.cor2 <- as.dist(1 - cor(plate.summary2 %>% select(-survival, -available)))
plate.clust2 <- hclust(plate.cor2)
plate.groups2 <- cutree(plate.clust2, k=16)
plate.PC2 <- 1:max(plate.groups2) %>% sapply(function(clust){
  sub.feats <- plate.summary2[names(plate.groups2[plate.groups2 == clust])] %>% prcomp()
  sub.feats$x[,'PC1']
}) %>% as.data.frame() %>%
  mutate(survival=plate.summary2$survival, available=plate.summary2$available)

cox_model <- coxph(Surv(survival, available) ~ ., data = plate.PC)
summary(cox_model)
ggplot(plate.PC, aes(survival, V15)) + geom_point(size=3) +
  geom_smooth(method='lm') + ylim(c(-2, 2))
ggplot(plate.PC, aes(survival, V4)) + geom_point(size=3) +
  geom_smooth(method='lm') + ylim(c(-2, 2))

cox_model2 <- coxph(Surv(survival, available) ~ ., data = plate.PC2)
summary(cox_model2)

names(plate.groups2)[plate.groups2 == 1]
names(plate.groups2)[plate.groups2 == 2]
names(plate.groups2)[plate.groups2 == 5]
names(plate.groups2)[plate.groups2 == 12]

cox_surv <- survfit(cox_model2, newdata = plate.PC2, type = "kaplan-meier", conf.int = FALSE)
plot(cox_surv)

ggplot(plate.PC2, aes(survival, V1)) + geom_point(size=3) +
  geom_smooth(method='lm') + ylim(c(-2, 2))
ggplot(plate.PC2, aes(survival, V2)) + geom_point(size=3) +
  geom_smooth(method='lm') + ylim(c(-2, 2))
ggplot(plate.PC2, aes(survival, V5)) + geom_point(size=3) +
  geom_smooth(method='lm') + ylim(c(-2, 2))
ggplot(plate.PC2, aes(survival, V12)) + geom_point(size=3) +
  geom_smooth(method='lm') + ylim(c(-2, 2))




(my.grades %>% inner_join(death_date, by='case') %>% ggplot(aes(high_s+h_per, survival))) + geom_point(size=3) + geom_smooth(method='lm')
(my.grades %>% inner_join(death_date, by='case') %>% ggplot(aes(met_interval, high_s+h_per))) + geom_point(size=3) + geom_smooth(method='lm')


###############################################
{
  # Fit the Kaplan-Meier survival curve to your data
  survival_data <- death_date %>% mutate(available = !cencored) %>%
    inner_join(my.grades, by='case') %>% filter(domin_s != 3) ####### IMPORTANT: FILTER?
  km_fit <- survfit(
    Surv(survival, available) ~ domin_s,
    data = survival_data
  )
  
  # Plot the Kaplan-Meier survival curve
  plot(
    km_fit, main = "Kaplan-Meier Survival - Dominant Scores", xlab = "Time (months)",
    ylab = "Survival probability", col=c('red', 'blue')
  )
  legend("bottomleft", legend = levels(factor(survival_data$domin_s)), col = c('red', 'blue'), lty = 1)
  med_times <- lapply(1:2, function(x) {
    quantile(survival_data$survival[survival_data$domin_s == x], probs = 0.5, na.rm=T)
  })
  
  # Add vertical lines at the median survival times
  for (i in 1:length(med_times)) {
    abline(v = med_times[[i]], lty = 2, col = c('red', 'blue')[i])
  }
  
  
  # Calculate the log-rank test p-value
  survdiff_result <- survdiff(Surv(survival, available) ~ domin_s, data = survival_data)
  p_value <- round(1 - pchisq(survdiff_result$chisq, length(survdiff_result$n) - 1), 4)
  
  # Add the p-value to the plot
  text(60, 0.9, paste0("Log-rank test p-value = ", p_value))
}
{
  # Fit the Kaplan-Meier survival curve to your data
  survival_data <- death_date %>% mutate(available = !cencored) %>%
    inner_join(my.grades, by='case') %>% filter(high_s != 1) ####### IMPORTANT: FILTER?
  km_fit <- survfit(
    Surv(survival, available) ~ high_s,
    data = survival_data
  )
  
  # Plot the Kaplan-Meier survival curve
  plot(
    km_fit, main = "Kaplan-Meier Survival - Highest Scores", xlab = "Time (months)",
    ylab = "Survival probability", col=c('red', 'blue')
  )
  legend("bottomleft", legend = levels(factor(survival_data$high_s)), col = c('red', 'blue'), lty = 1)
  med_times <- lapply(2:3, function(x) {
    quantile(survival_data$survival[survival_data$high_s == x], probs = 0.5, na.rm=T)
  })
  
  # Add vertical lines at the median survival times
  for (i in 1:length(med_times)) {
    abline(v = med_times[[i]], lty = 2, col = c('red', 'blue')[i])
  }
  
  
  # Calculate the log-rank test p-value
  survdiff_result <- survdiff(Surv(survival, available) ~ high_s, data = survival_data)
  p_value <- round(1 - pchisq(survdiff_result$chisq, length(survdiff_result$n) - 1), 4)
  
  # Add the p-value to the plot
  text(60, 0.9, paste0("Log-rank test p-value = ", p_value))
}






