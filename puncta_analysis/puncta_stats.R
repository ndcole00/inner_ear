library(ggpubr)
library(readr)
library(rstatix)
library(gridExtra)


dataFile <- '/home/nick/Documents/Analysis/R/Puncta_analysis/Pilot/Puncta_datatable.csv'
punctaData <- read_csv(dataFile, name_repair = "unique_quiet")

## P(PUNCTA) STATS

# test pre-synaptic P
pre_one.way <- punctaData %>%
  group_by(Frequency) %>%
  anova_test(dv=Presynaptic_P, wid=...1,between=Condition) %>%
  get_anova_table() %>%
  adjust_pvalue(method='bonferroni')
pre_one.way
pre.pwc <- punctaData %>%
  group_by(Frequency) %>%
  games_howell_test(Presynaptic_P~Condition)

# test post-synaptic P
post_one.way <- punctaData %>%
  group_by(Frequency) %>%
  anova_test(dv=Postsynaptic_P, wid=...1,between=Condition) %>%
  get_anova_table() %>%
  adjust_pvalue(method='bonferroni')
post_one.way
post.pwc <- punctaData %>%
  group_by(Frequency) %>%
  games_howell_test(Postsynaptic_P~Condition)

# test co-localised P
coloc_one.way <- punctaData %>%
  group_by(Frequency) %>%
  anova_test(dv=Colocalised_P, wid=...1,between=Condition) %>%
  get_anova_table() %>%
  adjust_pvalue(method='bonferroni')
coloc_one.way
coloc.pwc <- punctaData %>%
  group_by(Frequency) %>%
  games_howell_test(Colocalised_P~Condition)

punctaData = tibble(punctaData,show_col_types = FALSE)


## PUNCTA AREA
plot_list = list()
p <- ggplot(punctaData, aes(x = factor(Frequency), 
                          y = Mean_area_presynaptic,
                          fill = factor(Condition))) +
  geom_boxplot() +
  scale_fill_manual(values=c("#ef8a62","#67a9cf")) +
  geom_point(position=position_dodge(width=0.75),
             aes(group=factor(Condition))) +
  labs(title = 'Mean area of presynaptic puncta', 
       x = "Frequency (kHz)", y = "Presynaptic puncta area (um^2)") +
  stat_compare_means(aes(group = factor(Condition)),
                     method = 'kruskal.test',
                     label = "p.format",
                     label.y = max(punctaData$Mean_area_presynaptic)+0.02,
                     size=3) + 
  ylim(min(punctaData$Mean_area_presynaptic),max(punctaData$Mean_area_presynaptic)+0.06) +
  theme_minimal()
plot_list[[1]] <- p

p <- ggplot(punctaData, aes(x = factor(Frequency), 
                            y = Mean_area_postsynaptic,
                            fill = factor(Condition))) +
  geom_boxplot() +
  scale_fill_manual(values=c("#ef8a62","#67a9cf")) +
  geom_point(position=position_dodge(width=0.75),
             aes(group=factor(Condition))) +
  labs(title = 'Mean area of postsynaptic puncta', 
       x = "Frequency (kHz)", y = "Postsynaptic puncta area (um^2)") +
  stat_compare_means(aes(group = factor(Condition)),
                     method = 'kruskal.test',
                     label = "p.format",
                     label.y = max(punctaData$Mean_area_postsynaptic)+0.02,
                     size=3) + 
  ylim(min(punctaData$Mean_area_postsynaptic),max(punctaData$Mean_area_postsynaptic)+0.06) +
  theme_minimal()
plot_list[[2]] <- p
image = grid.arrange(grobs = plot_list, ncol = 1)

# ANOVA of areas
pre_area_one.way <- punctaData %>%
  group_by(Frequency) %>%
  anova_test(dv=Mean_area_presynaptic, wid=...1,between=Condition) %>%
  get_anova_table() %>%
  adjust_pvalue(method='bonferroni')
pre_area_one.way
pre_area.pwc <- punctaData %>%
  group_by(Frequency) %>%
  games_howell_test(Mean_area_presynaptic~Condition)

post_area.way <- punctaData %>%
  group_by(Frequency) %>%
  anova_test(dv=Mean_area_postsynaptic, wid=...1,between=Condition) %>%
  get_anova_table() %>%
  adjust_pvalue(method='bonferroni')
post_area.way
post_area.pwc <- punctaData %>%
  group_by(Frequency) %>%
  games_howell_test(Mean_area_postsynaptic~Condition)


