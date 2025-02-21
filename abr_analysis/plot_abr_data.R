rm(list = ls()) # clear workspace

# Check if required packages are installed, if not install them
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("pracma")) install.packages("pracma")

dpoaeDirectory <- '/home/nc/Documents/Analysis/DPOAEs_dataTable_allMice.csv'
clickDirectory <- '/home/nc/Documents/Analysis/click_dataTable_allMice.csv'
ptDirectory <- '/home/nc/Documents/Analysis/pt_dataTable_allMice.csv'
saveDirectory <- '/home/nc/Documents/Analysis/R/Plots/Valentin/'

# load data
dpoaeData <- read.csv(dpoaeDirectory)
clickData <- read.csv(clickDirectory)
ptData <- read.csv(ptDirectory)

# set parameters
distinct_dbs <- unique(clickData$dB.Level)
#distinct_freqs <- sort(unique(ptData$Frequency..Hz.))
distinct_freqs <- as.numeric(c("8000","16000","24000","32000"))
timepoints = c("PE", "D1", "D3", "D7", "D14")
if ("Condition" %in% colnames(ptData))
  {conditions = unique(ptData$Conditions)
}else{
  {ptData$Condition <- repmat(1,nrow(ptData),1)[,1]
  clickData$Condition <- repmat(1,nrow(clickData),1)[,1]
  dpoaeData$Condition <- repmat(1,nrow(dpoaeData),1)[,1]
  conditions = 1
  }

#conditions = "Trauma" # if using Valentin's only
control_mice = unique(clickData$Mouse.Name[clickData$Condition=='Control'])
trauma_mice = unique(clickData$Mouse.Name[clickData$Condition=='Trauma'])
# range of greyscale colours between white and black for individual mice
control_mice_cols = repmat(t(linspace(0.05,0.95,length(control_mice))),3,1)
trauma_mice_cols = repmat(t(linspace(0.05,0.95,length(trauma_mice))),3,1)
# set colours manually for timepoints - R prefers HEX
cols = c("#FDC086FF","#A6CEE3FF","#1F78B4FF","#B2DF8AFF","#33A02CFF")
mice <- unique(clickData$Mouse.Name)

# optional: only plot data recorded by one person
ptData <- subset(ptData,Recorded_by=="Valentin")
clickData <- subset(clickData,Recorded_by=="Valentin")

# account for both ear recordings in Valentin's data
for (mm in (1:length(mice))){
  for (db in (1:length(distinct_dbs))){
    for (tt in (1:length(timepoints))){
      for (ff in (1:length(frequency))){
       if (length(unique(ptData$Ear[ptData$Mouse.Name==mice[mm]&ptData$Frequency..Hz.==distinct_freqs[ff]&ptData$Timepoint==timepoints[tt]&ptData$dB.Level==distinct_dbs[db]&ptData$Ear!="remove"])) > 1){
          # Average values across both ears
          ptData$Wave.I.amplitude..P1.T1...μV.[ptData$Mouse.Name==mice[mm]&ptData$Frequency..Hz.==distinct_freqs[ff]&ptData$Timepoint==timepoints[tt]&ptData$dB.Level==distinct_dbs[db]&ptData$Ear=="OG"]=mean(ptData$Wave.I.amplitude..P1.T1...μV.[ptData$Mouse.Name==mice[mm]&ptData$Frequency..Hz.==distinct_freqs[ff]&ptData$Timepoint==timepoints[tt]&ptData$dB.Level==distinct_dbs[db]&ptData$Ear=="OG"],ptData$Wave.I.amplitude..P1.T1...μV.[ptData$Mouse.Name==mice[mm]&ptData$Frequency..Hz.==distinct_freqs[ff]&ptData$Timepoint==timepoints[tt]&ptData$dB.Level==distinct_dbs[db]&ptData$Ear=="OD"],na.rm=TRUE)
          ptData$Latency.to.First.Peak..ms.[ptData$Mouse.Name==mice[mm]&ptData$Frequency..Hz.==distinct_freqs[ff]&ptData$Timepoint==timepoints[tt]&ptData$dB.Level==distinct_dbs[db]&ptData$Ear=="OG"]=mean(ptData$Latency.to.First.Peak..ms.[ptData$Mouse.Name==mice[mm]&ptData$Frequency..Hz.==distinct_freqs[ff]&ptData$Timepoint==timepoints[tt]&ptData$dB.Level==distinct_dbs[db]&ptData$Ear=="OG"],ptData$Latency.to.First.Peak..ms.[ptData$Mouse.Name==mice[mm]&ptData$Frequency..Hz.==distinct_freqs[ff]&ptData$Timepoint==timepoints[tt]&ptData$dB.Level==distinct_dbs[db]&ptData$Ear=="OD"],na.rm=TRUE)
          ptData$Amplitude.Ratio..Peak1.Peak4.[ptData$Mouse.Name==mice[mm]&ptData$Frequency..Hz.==distinct_freqs[ff]&ptData$Timepoint==timepoints[tt]&ptData$dB.Level==distinct_dbs[db]&ptData$Ear=="OG"]=(ptData$Amplitude.Ratio..Peak1.Peak4.[ptData$Mouse.Name==mice[mm]&ptData$Frequency..Hz.==distinct_freqs[ff]&ptData$Timepoint==timepoints[tt]&ptData$dB.Level==distinct_dbs[db]&ptData$Ear=="OG"]+ptData$Amplitude.Ratio..Peak1.Peak4.[ptData$Mouse.Name==mice[mm]&ptData$Frequency..Hz.==distinct_freqs[ff]&ptData$Timepoint==timepoints[tt]&ptData$dB.Level==distinct_dbs[db]&ptData$Ear=="OD"])/2
          ptData$Estimated.Threshold[ptData$Mouse.Name==mice[mm]&ptData$Frequency..Hz.==distinct_freqs[ff]&ptData$Timepoint==timepoints[tt]&ptData$dB.Level==distinct_dbs[db]&ptData$Ear=="OG"]=mean(ptData$Estimated.Threshold[ptData$Mouse.Name==mice[mm]&ptData$Frequency..Hz.==distinct_freqs[ff]&ptData$Timepoint==timepoints[tt]&ptData$dB.Level==distinct_dbs[db]&ptData$Ear=="OG"],ptData$Estimated.Threshold[ptData$Mouse.Name==mice[mm]&ptData$Frequency..Hz.==distinct_freqs[ff]&ptData$Timepoint==timepoints[tt]&ptData$dB.Level==distinct_dbs[db]&ptData$Ear=="OD"],na.rm=TRUE)
          ptData$Ear[ptData$Mouse.Name==mice[mm]&ptData$Frequency..Hz.==distinct_freqs[ff]&ptData$Timepoint==timepoints[tt]&ptData$dB.Level==distinct_dbs[db]&ptData$Ear=="OD"] = "remove"
         }
      }
    # Do the same for click sessions
    if (length(unique(clickData$Ear[clickData$Mouse.Name==mice[mm]&clickData$Timepoint==timepoints[tt]&clickData$dB.Level==distinct_dbs[db]&clickData$Ear!="remove"])) > 1){
        clickData$Wave.I.amplitude..P1.T1...μV.[clickData$Mouse.Name==mice[mm]&clickData$Timepoint==timepoints[tt]&clickData$dB.Level==distinct_dbs[db]&clickData$Ear=="OG"]=mean(clickData$Wave.I.amplitude..P1.T1...μV.[clickData$Mouse.Name==mice[mm]&clickData$Timepoint==timepoints[tt]&clickData$dB.Level==distinct_dbs[db]&clickData$Ear=="OG"],clickData$Wave.I.amplitude..P1.T1...μV.[clickData$Mouse.Name==mice[mm]&clickData$Timepoint==timepoints[tt]&clickData$dB.Level==distinct_dbs[db]&clickData$Ear=="OD"],na.rm=TRUE)
        clickData$Latency.to.First.Peak..ms.[clickData$Mouse.Name==mice[mm]&clickData$Timepoint==timepoints[tt]&clickData$dB.Level==distinct_dbs[db]&clickData$Ear=="OG"]=mean(clickData$Latency.to.First.Peak..ms.[clickData$Mouse.Name==mice[mm]&clickData$Timepoint==timepoints[tt]&clickData$dB.Level==distinct_dbs[db]&clickData$Ear=="OG"],clickData$Latency.to.First.Peak..ms.[clickData$Mouse.Name==mice[mm]&clickData$Timepoint==timepoints[tt]&clickData$dB.Level==distinct_dbs[db]&clickData$Ear=="OD"],na.rm=TRUE)
        clickData$Amplitude.Ratio..Peak1.Peak4.[clickData$Mouse.Name==mice[mm]&clickData$Timepoint==timepoints[tt]&clickData$dB.Level==distinct_dbs[db]&clickData$Ear=="OG"]=(clickData$Amplitude.Ratio..Peak1.Peak4.[clickData$Mouse.Name==mice[mm]&clickData$Timepoint==timepoints[tt]&clickData$dB.Level==distinct_dbs[db]&clickData$Ear=="OG"]+clickData$Amplitude.Ratio..Peak1.Peak4.[clickData$Mouse.Name==mice[mm]&clickData$Timepoint==timepoints[tt]&clickData$dB.Level==distinct_dbs[db]&clickData$Ear=="OD"])/2
        clickData$Estimated.Threshold[clickData$Mouse.Name==mice[mm]&clickData$Timepoint==timepoints[tt]&clickData$dB.Level==distinct_dbs[db]&clickData$Ear=="OG"]=mean(clickData$Estimated.Threshold[clickData$Mouse.Name==mice[mm]&clickData$Timepoint==timepoints[tt]&clickData$dB.Level==distinct_dbs[db]&clickData$Ear=="OG"],clickData$Estimated.Threshold[clickData$Mouse.Name==mice[mm]&clickData$Timepoint==timepoints[tt]&clickData$dB.Level==distinct_dbs[db]&clickData$Ear=="OD"],na.rm=TRUE)
       clickData$Ear[clickData$Mouse.Name==mice[mm]&clickData$Timepoint==timepoints[tt]&clickData$dB.Level==distinct_dbs[db]&clickData$Ear=="OD"] = "remove"
    }
    }
  }
}
# Then remove these extra sessions
ptData <- subset(ptData,Ear!="remove")
clickData <- subset(clickData,Ear!="remove")

data = ptData[ptData$dB.Level==max(distinct_dbs),]
ct_data = clickData[clickData$dB.Level==max(distinct_dbs),]

# plot thresholds over time (not adjusted)
for (cond in(1:length(conditions))){
  yData <- array(numeric(),c(5,length(distinct_freqs)))
  yErr <- array(numeric(),c(5,length(distinct_freqs))) 
  for (tt in(1:(length(timepoints)))){
    for (ff in(1:length(distinct_freqs))){
      yErr[tt,ff]=std_err(na.omit(as.numeric(data$Estimated.Threshold[data$Condition==conditions[cond]&data$Frequency..Hz.==distinct_freqs[ff]&data$Timepoint==timepoints[tt]])))
      yData[tt,ff]=mean(na.omit(as.numeric(data$Estimated.Threshold[data$Condition==conditions[cond]&data$Frequency..Hz.==distinct_freqs[ff]&data$Timepoint==timepoints[tt]]),rm.na=TRUE))
    }
    if (tt==1){par(fig=c(0.1,0.9,0,1),new=FALSE)}else{par(fig=c(0.1,0.9,0,1),new=TRUE)} # only overwrite if first timepoint
    plot(distinct_freqs,yData[tt,],xlab='Frequencies (Hz)',type="b",ylab='Esimated hearing threshold (dB)',main=conditions[cond],col=cols[tt],ylim=c(0,90))
    arrows(distinct_freqs, yData[tt,]-yErr[tt,]/2, distinct_freqs, yData[tt,]+yErr[tt,]/2, length=0.05, angle=90, code=3,col=cols[tt])
  }
  legend(min(distinct_freqs),85,legend=timepoints,col=cols,lty=1,cex=0.8)
  for (tt in(1:(length(timepoints)))){
    yErr=std_err(na.omit(as.numeric(ct_data$Estimated.Threshold[ct_data$Condition==conditions[cond]&ct_data$Timepoint==timepoints[tt]])))
    yData=mean(na.omit(as.numeric(ct_data$Estimated.Threshold[ct_data$Condition==conditions[cond]&ct_data$Timepoint==timepoints[tt]]),rm.na=TRUE))
    par(fig=c(0.8,1,0,1),new=TRUE)
    plot(1,yData,ylab='',ann=FALSE,xaxt='n',xlab='Clicks',yaxt='n',type='b',col=cols[tt],ylim=c(0,90))
    arrows(1, yData-yErr/2, 1, yData+yErr/2, length=0.05, angle=90, code=3,col=cols[tt])
  }
  dev.print(pdf, sprintf('%s%s_estimated_thresholds.pdf',saveDirectory,conditions[cond]))
}


# calculate threshold change across mice
for (mm in (1:length(mice))){
  for (ff in (1:length(distinct_freqs))){
    threshold=data$Estimated.Threshold[data$Mouse.Name==mice[mm]&data$Frequency..Hz.==distinct_freqs[ff]&data$Timepoint==timepoints[1]]
    if (length(threshold)==0){ # in case of missing values e.g. frequencies in Valentin's data
      data$Estimated.Threshold[data$Mouse.Name==mice[mm]&data$Frequency..Hz.==distinct_freqs[ff]] = NA
    }else{
        data$Estimated.Threshold[data$Mouse.Name==mice[mm]&data$Frequency..Hz.==distinct_freqs[ff]] = as.numeric(data$Estimated.Threshold[data$Mouse.Name==mice[mm]&data$Frequency..Hz.==distinct_freqs[ff]]) - as.numeric(threshold)
    }}
  # do the same for click sessions
  threshold=ct_data$Estimated.Threshold[ct_data$Mouse.Name==mice[mm]&ct_data$Timepoint==timepoints[1]]
  ct_data$Estimated.Threshold[ct_data$Mouse.Name==mice[mm]] = ct_data$Estimated.Threshold[ct_data$Mouse.Name==mice[mm]] - threshold
  }

# plot wave 1 amplitudes over time
for (cond in(1:length(conditions))){
  yData <- array(numeric(),c(5,length(distinct_freqs)))
  yErr <- array(numeric(),c(5,length(distinct_freqs))) 
  for (tt in(1:(length(timepoints)))){
    for (ff in(1:length(distinct_freqs))){
      yErr[tt,ff]=std_err(na.omit(as.numeric(data$Wave.I.amplitude..P1.T1...μV.[data$Condition==conditions[cond]&data$Frequency..Hz.==distinct_freqs[ff]&data$Timepoint==timepoints[tt]])))
      yData[tt,ff]=mean(na.omit(as.numeric(data$Wave.I.amplitude..P1.T1...μV.[data$Condition==conditions[cond]&data$Frequency..Hz.==distinct_freqs[ff]&data$Timepoint==timepoints[tt]]),rm.na=TRUE))
    }
    if (tt==1){par(fig=c(0.1,0.9,0,1),new=FALSE)}else{par(fig=c(0.1,0.9,0,1),new=TRUE)} # only overwrite if first timepoint
    plot(distinct_freqs,yData[tt,],xlab='Frequencies (Hz)',type="b",ylab='Wave 1 amplitude (μV)',main=conditions[cond],col=cols[tt],ylim=c(0,2.5))
    arrows(distinct_freqs, yData[tt,]-yErr[tt,]/2, distinct_freqs, yData[tt,]+yErr[tt,]/2, length=0.05, angle=90, code=3,col=cols[tt])
  }
  legend(30000,2.9,legend=timepoints,col=cols,lty=1,cex=0.8)
  for (tt in(1:(length(timepoints)))){
    yErr=std_err(na.omit(as.numeric(ct_data$Wave.I.amplitude..P1.T1...μV.[ct_data$Condition==conditions[cond]&ct_data$Timepoint==timepoints[tt]])))
    yData=mean(na.omit(as.numeric(ct_data$Wave.I.amplitude..P1.T1...μV.[ct_data$Condition==conditions[cond]&ct_data$Timepoint==timepoints[tt]]),rm.na=TRUE))
    par(fig=c(0.8,1,0,1),new=TRUE)
    plot(1,yData,ylab='',ann=FALSE,xaxt='n',xlab='Clicks',yaxt='n',type='b',col=cols[tt],ylim=c(0,2.5))
    arrows(1, yData-yErr/2, 1, yData+yErr/2, length=0.05, angle=90, code=3,col=cols[tt])
  }
  dev.print(pdf, sprintf('%s%s_wave1_amplitude.pdf',saveDirectory,conditions[cond]))
}

# plot latency to first peak over time
for (cond in(1:length(conditions))){
  yData <- array(numeric(),c(5,length(distinct_freqs)))
  yErr <- array(numeric(),c(5,length(distinct_freqs))) 
  for (tt in(1:(length(timepoints)))){
    for (ff in(1:length(distinct_freqs))){
      yErr[tt,ff]=std_err(na.omit(as.numeric(data$Latency.to.First.Peak..ms.[data$Condition==conditions[cond]&data$Frequency..Hz.==distinct_freqs[ff]&data$Timepoint==timepoints[tt]])))
      yData[tt,ff]=mean(na.omit(as.numeric(data$Latency.to.First.Peak..ms.[data$Condition==conditions[cond]&data$Frequency..Hz.==distinct_freqs[ff]&data$Timepoint==timepoints[tt]]),rm.na=TRUE))
    }
    if (tt==1){par(fig=c(0.1,0.9,0,1),new=FALSE)}else{par(fig=c(0.1,0.9,0,1),new=TRUE)} # only overwrite if first timepoint
    plot(distinct_freqs,yData[tt,],xlab='Frequencies (Hz)',type="b",ylab='Latency to first peak (ms)',main=conditions[cond],col=cols[tt],ylim=c(1,3))
    arrows(distinct_freqs, yData[tt,]-yErr[tt,]/2, distinct_freqs, yData[tt,]+yErr[tt,]/2, length=0.05, angle=90, code=3,col=cols[tt])
  }
  legend(30000,3.45,legend=timepoints,col=cols,lty=1,cex=0.8)
  for (tt in(1:(length(timepoints)))){
    yErr=std_err(na.omit(as.numeric(ct_data$Latency.to.First.Peak..ms.[ct_data$Condition==conditions[cond]&ct_data$Timepoint==timepoints[tt]])))
    yData=mean(na.omit(as.numeric(ct_data$Latency.to.First.Peak..ms.[ct_data$Condition==conditions[cond]&ct_data$Timepoint==timepoints[tt]]),rm.na=TRUE))
    par(fig=c(0.8,1,0,1),new=TRUE)
    plot(1,yData,ylab='',ann=FALSE,xaxt='n',xlab='Clicks',yaxt='n',type='b',col=cols[tt],ylim=c(1,3))
    arrows(1, yData-yErr/2, 1, yData+yErr/2, length=0.05, angle=90, code=3,col=cols[tt])
  }
  dev.print(pdf, sprintf('%s%s_latency_to_first_peak.pdf',saveDirectory,conditions[cond]))
}


cols <- cols[2:5] # remove PE colour

# plot hearing threshold changes over time
for (cond in(1:length(conditions))){
  yData <- array(numeric(),c(4,length(distinct_freqs)))
  yErr <- array(numeric(),c(4,length(distinct_freqs))) 
  for (tt in(1:(length(timepoints)-1))){
    for (ff in(1:length(distinct_freqs))){
      yErr[tt,ff]=std_err(na.omit(as.numeric(data$Estimated.Threshold[data$Condition==conditions[cond]&data$Frequency..Hz.==distinct_freqs[ff]&data$Timepoint==timepoints[tt+1]])))
      yData[tt,ff]=mean(na.omit(as.numeric(data$Estimated.Threshold[data$Condition==conditions[cond]&data$Frequency..Hz.==distinct_freqs[ff]&data$Timepoint==timepoints[tt+1]]),rm.na=TRUE))
    }
    if (tt==1){par(fig=c(0.1,0.9,0,1),new=FALSE)}else{par(fig=c(0.1,0.9,0,1),new=TRUE)} # only overwrite if first timepoint
    plot(distinct_freqs,yData[tt,],xlab='Frequencies (Hz)',type="b",ylab='Threshold change (dB)',main=conditions[cond],col=cols[tt],ylim=c(-15,50))
    arrows(distinct_freqs, yData[tt,]-yErr[tt,]/2, distinct_freqs, yData[tt,]+yErr[tt,]/2, length=0.05, angle=90, code=3,col=cols[tt])
  }
  legend(min(distinct_freqs),45,legend=timepoints[2:5],col=cols,lty=1,cex=0.8)
  for (tt in(1:(length(timepoints)-1))){
    yErr=std_err(na.omit(as.numeric(ct_data$Estimated.Threshold[ct_data$Condition==conditions[cond]&ct_data$Timepoint==timepoints[tt+1]])))
    yData=mean(na.omit(as.numeric(ct_data$Estimated.Threshold[ct_data$Condition==conditions[cond]&ct_data$Timepoint==timepoints[tt+1]]),rm.na=TRUE))
    par(fig=c(0.8,1,0,1),new=TRUE)
    plot(1,yData,ylab='',ann=FALSE,xaxt='n',xlab='Clicks',yaxt='n',type='b',col=cols[tt],ylim=c(-15,50))
    arrows(1, yData-yErr/2, 1, yData+yErr/2, length=0.05, angle=90, code=3,col=cols[tt])
  }
  dev.print(pdf, sprintf('%s%s_threshold_change.pdf',saveDirectory,conditions[cond]))
}
  





