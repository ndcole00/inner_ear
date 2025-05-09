# Function for analysing waves from ABR data, over days

if (!require("gridExtra")) install.packages("gridExtra")
if (!require("ggpubr")) install.packages("ggpubr")
if (!require("tidyr")) install.packages("tidyr")
if (!require("rstatix")) install.packages("rstatix")


clickDirectory = '/home/nick/Documents/Analysis/click_dataTable_allCond.csv'
ptDirectory = '/home/nick/Documents/Analysis/pt_dataTable_allCond.csv'
saveDirectory = '/home/nick/Documents/Analysis/R/abr_plots/'

# replace with your timepoints, in order
timepoints = c("PE","D1","D3","D7","D14")
# colours to use for each of these timepoints (must be HEX)
timeColours = c("#969696","#a1dab4","#41b6c4","#2c7fb8","#253494")
# set range of dBs to plot
dB_range = seq(from = 50, to = 90, by = 5)

## CLICK SESSIONS ###
clickData = read.csv(clickDirectory)
# sort by date and db level
clickData = clickData[order(clickData$Date,clickData$dB.Level),]
mice = unique(clickData$Mouse.Name)
# filter out unwanted dB levels
clickData <- subset(clickData,dB.Level>=min(dB_range)&dB.Level<=max(dB_range))
clickData <- data.frame(clickData)
# rename wave 1 column
colnames(clickData)[10] <- "wave1_amp"
all_mice <- list()
#ymax = max(clickData$wave1_amp)
ymax = 10

for (mm in (1:length(mice))){
  data = subset(clickData,Mouse.Name==mice[mm])
  p <- ggline(data,
         x='dB.Level',
         y='wave1_amp',
         palette="jco",
         #group='Timepoint',
         color='Timepoint',
         title=mice[mm],
         xlab="dB level",
         ylab="Wave 1 amplitude (μV)") +
    ylim(0,ymax)
  all_mice[[mm]] <- p
}
image = grid.arrange(grobs = all_mice, ncol = 2)
ggsave(file=sprintf("%sAll_mice_click_wave1_amplitude.svg",saveDirectory),
       plot=image,width=10,height=8)

# basic stats: effect of trauma at each time point
click_one.way <- clickData %>%
  group_by(factor(Timepoint)) %>%
  anova_test(dv = wave1_amp, wid = X, between = Condition) %>%
  get_anova_table() %>%
  adjust_pvalue(method = "bonferroni")
click_one.way

# test between timepoints of trauma mice only
traumaClickData = clickData[clickData$Condition=='Trauma',]
# effect of time for each group, grouped by frequency
trauma_click_one.way <- traumaClickData %>%
  anova_test(dv = wave1_amp, wid = X, between = Timepoint) %>%
  get_anova_table() %>%
  adjust_pvalue(method = "bonferroni")
trauma_click_one.way
trauma_click.pwc <- traumaClickData %>%
  games_howell_test(wave1_amp~Timepoint)

p <- ggline(clickData,
       x = "dB.Level",
       y = "wave1_amp", 
       add = c("mean_se", "jitter"),
       color='Timepoint',
       palette = "jco",
       title="Mean Wave 1 amplitudes",
       xlab="dB level",
       ylab="Wave 1 amplitude (μV)") +
  ylim(0,ymax)
ggsave(file=sprintf("%sMean_click_wave1_amplitude.svg",saveDirectory),
       plot=p,width=10,height=12)

## PURE TONE SESSIONS ##
ptData = read.csv(ptDirectory)
freqs = unique(ptData$Frequency..Hz.)
ptData = ptData[order(ptData$Date,ptData$dB.Level),]
mice = unique(ptData$Mouse.Name)
# filter out unwanted dB levels
ptData <- subset(ptData,dB.Level>=min(dB_range)&dB.Level<=max(dB_range))
ptData <- data.frame(ptData)
# rename wave 1 column
colnames(ptData)[10] <- "wave1_amp"
all_mice <- list()
#ymax = max(ptData$wave1_amp)
ymax = 3

for (freq in (1:length(freqs))){
  for (mm in (1:length(mice))){
    data = subset(ptData,Mouse.Name==mice[mm]&Frequency..Hz.==freqs[freq])
    p <- ggline(data,
                x='dB.Level',
                y='wave1_amp',
                palette="jco",
                #group='Timepoint',
                color='Timepoint',
                title=mice[mm],
                xlab="dB level",
                ylab="Wave 1 amplitude (μV)") +
      ylim(0,ymax)
    all_mice[[mm]] <- p
  }
  image = grid.arrange(grobs = all_mice, ncol = 2,title=sprintf("%sHz",freqs[freq]))
  ggsave(file=sprintf("%sAll_mice_click_wave1_amplitude_%sHz.svg",saveDirectory,freqs[freq]),
         plot=image,width=10,height=8)
  
  data = subset(ptData,Frequency..Hz.==freqs[freq])
  p <- ggline(data,
              x = "dB.Level",
              y = "wave1_amp", 
              add = c("mean_se", "jitter"),
              color='Timepoint',
              palette = "jco",
              title=sprintf("Mean Wave 1 amplitudes - %sHz",freqs[freq]),
              xlab="dB level",
              ylab="Wave 1 amplitude (μV)") +
    ylim(0,ymax)
  ggsave(file=sprintf("%sMean_click_wave1_amplitude_%sHz.svg",saveDirectory,freqs[freq]),
         plot=p,width=10,height=12)
  
}

# basic stats: effect of trauma at each time point
pt_one.way <- ptData %>%
  group_by(factor(Timepoint)) %>%
  anova_test(dv = wave1_amp, wid = X, between = Condition) %>%
  get_anova_table() %>%
  adjust_pvalue(method = "bonferroni")
pt_one.way

# test between timepoints of trauma mice only
traumaPTData = ptData[ptData$Condition=='Trauma',]
# effect of time for each group, grouped by frequency
trauma_pt_one.way <- traumaPTData %>%
  anova_test(dv = wave1_amp, wid = X, between = Timepoint) %>%
  get_anova_table() %>%
  adjust_pvalue(method = "bonferroni")
trauma_pt_one.way
trauma_pt.pwc <- traumaPTData %>%
  group_by(Frequency..Hz.) %>%
  games_howell_test(wave1_amp~Timepoint)