  library(tidyverse)
  library(ggpubr)
  library(rstatix)
  
  dpoaeDirectory <- '/home/nc/Documents/Analysis/Preprocessed_ABR_data/DPOAE_data_threshold.csv'
  clickDirectory <- '/home/nc/Documents/Analysis/Preprocessed_ABR_data/Click_data.csv'
  ptDirectory <- '/home/nc/Documents/Analysis/Preprocessed_ABR_data/Pure_tone_data.csv'
  saveDirectory <- '/home/nc/Documents/Analysis/R/Plots/'
  
  legendOn = TRUE # whether to plot legend
  
  # load data
  dpoaeData <- read.csv(dpoaeDirectory)
  clickData <- read.csv(clickDirectory)
  ptData <- read.csv(ptDirectory)
  
  distinct_dbs = unique(ptData$dB.Level)
  timepoints = c("PE","D1","D3","D7","D14")
  conditions = unique(ptData$Condition)
  mice = unique(ptData$Mouse.Name)
  
  # # Crop out any factors with missing values in Valentin's recordings
  # ptData <- subset(ptData,Frequency..Hz.!="4000")
  # clickData <- subset(clickData,Frequency..Hz.!="4000")
  # ptData <- subset(ptData,Frequency..Hz.!="48000")
  # clickData <- subset(clickData,Frequency..Hz.!="48000")
  # ptData <- subset(ptData,Mouse.Name!="3381")
  # clickData <- subset(clickData,Mouse.Name!="3381")
  # ptData <- subset(ptData,Mouse.Name!="CTRLF2")
  # clickData <- subset(clickData,Mouse.Name!="CTRLF2")
  # 
  # # And take off two of the trauma mice to give equal N
  # ptData <- subset(ptData,Mouse.Name!="3377")
  # clickData <- subset(clickData,Mouse.Name!="3377")
  # ptData <- subset(ptData,Mouse.Name!="TRAM0")
  # clickData <- subset(clickData,Mouse.Name!="TRAM0")

  # change timepoints into ordinal
  for (tt in (1:length(timepoints))){
    ptData$Timepoint[ptData$Timepoint==timepoints[tt]] = tt
    clickData$Timepoint[clickData$Timepoint==timepoints[tt]] = tt
  }
  # same with condition
  for (tt in (1:length(conditions))){
    ptData$Condition[ptData$Condition==conditions[tt]] = tt
    clickData$Condition[clickData$Condition==conditions[tt]] = tt
  }
  # and mice
  for (tt in (1:length(mice))){
    ptData$Mouse.Name[ptData$Mouse.Name==mice[tt]] = tt
    clickData$Mouse.Name[clickData$Mouse.Name==mice[tt]] = tt
  }
  
  # crop to just the max dBs
  ptData = ptData[ptData$dB.Level==max(distinct_dbs),]

  # convert the experimental variables into factors
  ptData$Condition = factor(ptData$Condition)
  clickData$Condition = factor(clickData$Condition)
  ptData$Timepoint = factor(ptData$Timepoint)
  clickData$Timepoint = factor(clickData$Timepoint)
  ptData$Frequency..Hz. = factor(ptData$Frequency..Hz.)
  ptData$X = factor(ptData$X)
  clickData$X = factor(clickData$X)
  
  # do some boxplots to check this works
  ggboxplot(ptData,x="Timepoint",y="Estimated.Threshold",color="Condition")
  ggboxplot(clickData,x="Timepoint",y="Estimated.Threshold",color="Condition")
  
  # test for outliers
  ptData %>% group_by(Timepoint, Condition, Frequency..Hz.) %>% identify_outliers(Estimated.Threshold)
  clickData %>% group_by(Timepoint, Condition) %>% identify_outliers(Estimated.Threshold)
  
  # effect of trauma at each time point and frequency
  one.way <- ptData %>%
    group_by(Timepoint,Frequency..Hz.) %>%
    anova_test(dv = Estimated.Threshold, wid = X, between = Condition) %>%
    get_anova_table() %>%
    adjust_pvalue(method = "bonferroni")
  one.way
  
  controlPTData = ptData[ptData$Condition==1,]
  # effect of time for each group, grouped by frequency
  ctrl_one.way <- controlPTData %>%
    group_by(Frequency..Hz.) %>%
    anova_test(dv = Estimated.Threshold, wid = X, between = Timepoint) %>%
    get_anova_table() %>%
    adjust_pvalue(method = "bonferroni")
  ctrl_one.way
  ctrl.pwc <- controlPTData %>%
    group_by(Frequency..Hz.) %>%
    games_howell_test(Estimated.Threshold~Timepoint)
  
  traumaPTData = ptData[ptData$Condition==2,]
  # effect of time for each group, grouped by frequency
  trauma_one.way <- traumaPTData %>%
    group_by(Frequency..Hz.) %>%
    anova_test(dv = Estimated.Threshold, wid = X, between = Timepoint) %>%
    get_anova_table() %>%
    adjust_pvalue(method = "bonferroni")
  trauma_one.way
  trauma.pwc <- traumaPTData %>%
    group_by(Frequency..Hz.) %>%
    games_howell_test(Estimated.Threshold~Timepoint)
  
  # CLICK DATA
  # effect of trauma at each time point
  one.way <- ptData %>%
    group_by(Timepoint) %>%
    anova_test(dv = Estimated.Threshold, wid = X, between = Condition) %>%
    get_anova_table() %>%
    adjust_pvalue(method = "bonferroni")
  one.way
  
  controlCLData = clickData[clickData$Condition==1,]
  # effect of time for each group, grouped by frequency
  ctrl_click_one.way <- controlCLData %>%
    anova_test(dv = Estimated.Threshold, wid = X, between = Timepoint) %>%
    get_anova_table() %>%
    adjust_pvalue(method = "bonferroni")
  ctrl_click_one.way
  ctrl_click.pwc <- controlPTData %>%
    group_by(Frequency..Hz.) %>%
    games_howell_test(Estimated.Threshold~Timepoint)
  
  traumaCLData = clickData[clickData$Condition==2,]
  # effect of time for each group, grouped by frequency
  trauma_click_one.way <- traumaCLData %>%
    anova_test(dv = Estimated.Threshold, wid = X, between = Timepoint) %>%
    get_anova_table() %>%
    adjust_pvalue(method = "bonferroni")
  trauma_click_one.way
  trauma_click.pwc <- traumaCLData %>%
    group_by(Frequency..Hz.) %>%
    games_howell_test(Estimated.Threshold~Timepoint)