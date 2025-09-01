  if (!require("dplyr")) install.packages("dplyr")
  if (!require("readr")) install.packages("readr")
  if (!require("tidyr")) install.packages("tidyr")
  if (!require("ggplot2")) install.packages("ggplot2")
  if (!require("gridExtra")) install.packages("gridExtra")
  if (!require("ggpubr")) install.packages("ggpubr")
  
  # Load required libraries
  library(dplyr)
  library(readr)
  library(tidyr)
  library(ggplot2)
  library(gridExtra)
  library(ggpubr)
  library(svglite)
  
  # Set file paths
  dataFile <- '/home/nick/Desktop/Control_media_KA/Puncta_counts/'
  cellCountFile <- '/home/nick/Desktop/Explant_cell_counts.csv'
  saveDirectory <- '/home/nick/Documents/Analysis/R/Puncta_analysis/Pilot/'
  condName <- "Control_media_KA"
  
  # Read cell count data
  cellCount <- read_csv(cellCountFile, show_col_types = FALSE, name_repair = "unique_quiet")
  # Remove other conditions
  cellCount <- subset(cellCount,Condition==condName)
  
  # Get list of files
  files <- list.files(dataFile, full.names = TRUE)
  
  # Initialize variables
  sample <- list()
  section <- list()
  condition <- list()
  nIHCs <- numeric()
  presynapticN_IHC <- numeric()
  postsynapticN_IHC <- numeric()
  nOHCs <- numeric()
  presynapticN_OHC <- numeric()
  postsynapticN_OHC <- numeric()
  meanAreaPre_IHC <- numeric()
  meanAreaPost_IHC <- numeric()
  meanAreaPre_OHC <- numeric()
  meanAreaPost_OHC <- numeric()
  colocalisedN_IHC <- numeric()
  colocalisedN_OHC <- numeric()
  areaData <- data.frame(Area <- numeric(),
                         Perim <- numeric(),
                         Condition <- list(),
                         Pre_post <- list())
  mm <- 0
  removeCount <- 0
  fileCount <- 0
  
  # Process each file
  for (ii in seq_along(files)) {
    filename <- strsplit(basename(files[ii]), '_')[[1]]
    
    # Update mouse counter every four files (assuming pairs of CtBP2 and Homer files, IHC and OHC)
    if (fileCount == 0) {
      mm <- mm + 1
      sample[mm] <- filename[1]
      section[mm] <- filename[2]
      condition[mm] <- condName
      
      # Find nCells from cell count file
      nIHCs[mm] <- cellCount$IHC_N[mm]
      nOHCs[mm] <- cellCount$OHC_N[mm]
    }
    
    # IHCs will come first
    if (nIHCs[mm] > 0) {
      if (sample[mm] == cellCount$Slice[mm]){
    if (grepl("IHC", files[ii])) {
      fileCount = fileCount + 1
    # Read data based on file type
    if (grepl("CtBP2", files[ii])) {
      data <- read_csv(files[ii], show_col_types = FALSE, name_repair = "unique_quiet")
      presynData <- data
    } else if (grepl("Homer", files[ii])) {
      data <- read_csv(files[ii], show_col_types = FALSE, name_repair = "unique_quiet")
    } else {
      next
    }
    
    # Remove overlapping ROIs
    roisToRemove <- rep(FALSE, nrow(data))
    
    for (roi in 1:nrow(data)) {
      current_slice <- data$Slice[roi]
      nextplane <- data %>% filter(Slice == current_slice + 1)
      
      if (nrow(nextplane) > 0) {
        # Find overlapping ROIs in next plane
        overlap <- nextplane %>%
          filter(X >= data$X[roi] - (data$Perim.[roi]/2),
                 X <= data$X[roi] + (data$Perim.[roi]/2),
                 Y >= data$Y[roi] - (data$Perim.[roi]/2),
                 Y <= data$Y[roi] + (data$Perim.[roi]/2))
        
        if (nrow(overlap) > 0) {
          roisToRemove[overlap$...1] = TRUE
          removeCount = removeCount + 1
        }
      }
    }
    
    # Remove marked ROIs
    data <- data[!roisToRemove, ]
    
    # Quality control
    circularity = 4 * pi * data$Area / data$Perim.^2
    data <- data[circularity>0.9, ] # remove all cells with circularity below 0.9
    
    tempData <- data.frame(Area <- numeric(),
                           Perim <- numeric(),
                           Condition <- list(),
                           Pre_post <- list())
    # Count puncta based on file type
    if (grepl("CtBP2", files[ii])) {
      presynapticN_IHC[mm] <- nrow(data)
      meanAreaPre_IHC[mm] <- mean(data$Area)
    } else if (grepl("Homer", files[ii])) {
      postsynapticN_IHC[mm] <- nrow(data)
      meanAreaPost_IHC[mm] <- mean(data$Area)
      count <- 0
      # Find co-localized synapses
      for (roi in 1:nrow(data)) {
        matches <- presynData %>%
          filter(X >= data$X[roi] - data$Perim.[roi],
                 X <= data$X[roi] + data$Perim.[roi],
                 Y >= data$Y[roi] - data$Perim.[roi],
                 Y <= data$Y[roi] + data$Perim.[roi],
                 Slice > data$Slice[roi] - 3,
                 Slice < data$Slice[roi] + 3)
        
        if (nrow(matches) > 0) {
          count <- count + 1
        }
      }
      colocalisedN_IHC[mm] <- count
    }
    }
      }
    }else{
      fileCount = fileCount + 2
      presynapticN_IHC[mm] <- NA
      postsynapticN_IHC[mm] <- NA
      colocalisedN_IHC[mm] <- NA
      meanAreaPre_IHC[mm] <- NA
      meanAreaPost_IHC[mm]  <- NA}
    
    # then do the same for OHCs
    if (nOHCs[mm] > 0) {
      if (sample[mm] == cellCount$Slice[mm]){
    if (grepl("OHC", files[ii])){
      fileCount = fileCount + 1
      # Read data based on file type
      if (grepl("CtBP2", files[ii])) {
        data <- read_csv(files[ii], show_col_types = FALSE, name_repair = "unique_quiet")
        presynData <- data
      } else if (grepl("Homer", files[ii])) {
        data <- read_csv(files[ii], show_col_types = FALSE, name_repair = "unique_quiet")
      } else {
        next
      }
      
      # Remove overlapping ROIs
      roisToRemove <- rep(FALSE, nrow(data))
      
      for (roi in 1:nrow(data)) {
        current_slice <- data$Slice[roi]
        nextplane <- data %>% filter(Slice == current_slice + 1)
        
        if (nrow(nextplane) > 0) {
          # Find overlapping ROIs in next plane
          overlap <- nextplane %>%
            filter(X >= data$X[roi] - (data$Perim.[roi]/2),
                   X <= data$X[roi] + (data$Perim.[roi]/2),
                   Y >= data$Y[roi] - (data$Perim.[roi]/2),
                   Y <= data$Y[roi] + (data$Perim.[roi]/2))
          
          if (nrow(overlap) > 0) {
            roisToRemove[overlap$...1] = TRUE
            removeCount = removeCount + 1
          }
        }
      }
      
      # Remove marked ROIs
      data <- data[!roisToRemove, ]
      
      # Quality control
      circularity = 4 * pi * data$Area / data$Perim.^2
      data <- data[circularity>0.9, ] # remove all cells with circularity below 0.9
      
      tempData <- data.frame(Area <- numeric(),
                             Perim <- numeric(),
                             Condition <- list(),
                             Pre_post <- list())
      # Count puncta based on file type
      if (grepl("CtBP2", files[ii])) {
        presynapticN_OHC[mm] <- nrow(data)
        meanAreaPre_OHC[mm] <- mean(data$Area)
      } else if (grepl("Homer", files[ii])) {
        postsynapticN_OHC[mm] <- nrow(data)
        meanAreaPost_OHC[mm] <- mean(data$Area)
        count <- 0
        # Find co-localized synapses
        for (roi in 1:nrow(data)) {
          matches <- presynData %>%
            filter(X >= data$X[roi] - data$Perim.[roi],
                   X <= data$X[roi] + data$Perim.[roi],
                   Y >= data$Y[roi] - data$Perim.[roi],
                   Y <= data$Y[roi] + data$Perim.[roi],
                   Slice > data$Slice[roi] - 3,
                   Slice < data$Slice[roi] + 3)
          
          if (nrow(matches) > 0) {
            count <- count + 1
          }
        }
      colocalisedN_OHC[mm] <- count
      }
    }
      }
    }else{
      fileCount = fileCount + 2
      presynapticN_OHC[mm] <- NA
      postsynapticN_OHC[mm] <- NA
      colocalisedN_OHC[mm] <- NA
      meanAreaPre_OHC[mm] <- NA
      meanAreaPost_OHC[mm]  <- NA}
    
    if (fileCount >= 4){
      fileCount <- 0
    }
  }


  # Create data frame
  data_df <- data.frame(
    Sample = unlist(sample),
    Section = unlist(section),
    Condition = unlist(condition),
    Presynaptic_N_IHC = presynapticN_IHC,
    Postsynaptic_N_IHC = postsynapticN_IHC,
    Colocalised_N_IHC = colocalisedN_IHC,
    Presynaptic_N_OHC = presynapticN_OHC,
    Postsynaptic_N_OHC = postsynapticN_OHC,
    Colocalised_N_OHC = colocalisedN_OHC,
    N_IHCs = nIHCs,
    N_OHCs = nOHCs,
    Mean_area_presynaptic_IHC = meanAreaPre_IHC,
    Mean_area_presynaptic_OHC = meanAreaPre_OHC,
    Mean_area_postsynaptic_IHC = meanAreaPost_IHC,
    Mean_area_postsynaptic_OHC = meanAreaPost_OHC
  )
  
  # Convert to tibble for better printing
  data_tbl <- as_tibble(data_df)

  write.csv(data_tbl,sprintf('%sProcessed_datatable.csv',dataFile)) # save the datatable
  
    # Plot 1: N puncta IHCs / N IHCs
    plot_list1 <- list()

    p <- ggplot(data_tbl, aes(x = factor(Section), 
                                y = Presynaptic_N_IHC / nIHCs)) +
        geom_boxplot() 
      plot_list1[[1]] <- p
      
      p <- ggplot(data_tbl, aes(x = factor(Section), 
                                y = Postsynaptic_N_IHC / nIHCs)) +
        geom_boxplot() 
      plot_list1[[2]] <- p

    image = grid.arrange(grobs = plot_list1, ncol = 1)
    ggsave(file=sprintf("%s%s_IHC_puncta_boxplots.svg",dataFile,condName),
           plot=image,width=10,height=8)
    
    # Plot 2: N puncta OHCs / N OHCs
    plot_list2 <- list()
    
    p <- ggplot(data_tbl, aes(x = factor(Section), 
                              y = Presynaptic_N_OHC / nOHCs)) +
      geom_boxplot() 
    plot_list2[[1]] <- p
    
    p <- ggplot(data_tbl, aes(x = factor(Section), 
                              y = Postsynaptic_N_OHC / nOHCs)) +
      geom_boxplot() 
    plot_list2[[2]] <- p
    
    image = grid.arrange(grobs = plot_list2, ncol = 1)
    ggsave(file=sprintf("%s%s_OHC_puncta_boxplots.svg",dataFile,condName),
           plot=image,width=10,height=8)