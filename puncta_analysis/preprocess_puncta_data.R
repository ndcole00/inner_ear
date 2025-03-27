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
  dataFile <- '/home/nick/Documents/Analysis/MATLAB/Puncta_analysis/Data'
  cellCountFile <- '/home/nick/Documents/Data/Blinded_data/Cell_counts.csv'
  saveDirectory <- '/home/nick/Documents/Analysis/R/Puncta_analysis/Pilot/'
  
  # Read cell count data
  cellCount <- read_csv(cellCountFile, col_names = c("Var1", "CellCount"), skip = 1, show_col_types = FALSE, name_repair = "unique_quiet")
  
  # Get list of files
  files <- list.files(dataFile, full.names = TRUE)
  
  # Initialize variables
  mouse <- list()
  frequency <- numeric()
  condition <- list()
  nCells <- numeric()
  presynapticN <- numeric()
  postsynapticN <- numeric()
  meanAreaPre <- numeric()
  meanAreaPost <- numeric()
  colocalisedN <- numeric()
  areaData <- data.frame(Area <- numeric(),
                         Perim <- numeric(),
                         Condition <- list(),
                         Pre_post <- list())
  mm <- 0
  removeCount <- 0
  # Process each file
  for (ii in seq_along(files)) {
    filename <- strsplit(basename(files[ii]), '_')[[1]]
    
    # Update mouse counter every two files (assuming pairs of CTBP and Homer files)
    if ((ii-2) %% 2 == 1) {
      mm <- mm + 1
      mouse[mm] <- filename[1]
      frequency[mm] <- as.numeric(gsub("kHz", "", filename[2]))
      condition[mm] <- gsub(".csv", "", filename[3])
      
      # Find nCells from cell count file
      countPattern <- paste0(mouse[mm], "_", frequency[mm], "kHz")
      countIdx <- grepl(countPattern, cellCount$Var1)
      nCells[mm] <- cellCount$CellCount[countIdx]
    }
    
    # Read data based on file type
    if (grepl("CTBP", files[ii])) {
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
    tempData <- data.frame(Area <- numeric(),
                           Perim <- numeric(),
                           Condition <- list(),
                           Pre_post <- list())
    
    # Count puncta based on file type
    if (grepl("CTBP", files[ii])) {
      presynapticN[mm] <- nrow(data)
      meanAreaPre[mm] <- mean(data$Area)
      if (grepl("CONTROL",files[ii])){
        tempData <- data.frame(Area=c(data$Area),
                               Perim=data$Perim.,
                               Condition=c(rep('CONTROL',nrow(data))),
                               Pre_post=c(rep('PRE',nrow(data))))
      }else if (grepl("TRAUMA",files[ii])){
        tempData <- data.frame(Area=data$Area,
                               Perim=data$Perim.,
                               Condition=rep("TRAUMA",nrow(data)),
                               Pre_post=rep("PRE",nrow(data)))
      }
    } else if (grepl("Homer", files[ii])) {
      postsynapticN[mm] <- nrow(data)
      meanAreaPost[mm] <- mean(data$Area)
      if (grepl("CONTROL",files[ii])){
        tempData <- data.frame(Area=data$Area,
                               Perim=data$Perim.,
                               Condition=rep("CONTROL",nrow(data)),
                               Pre_post=rep("POST",nrow(data)))
      }else if (grepl("TRAUMA",files[ii])){
        tempData <- data.frame(Area=data$Area,
                               Perim=data$Perim.,
                               Condition=rep("TRAUMA",nrow(data)),
                               Pre_post=rep("POST",nrow(data)))
      }
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
      colocalisedN[mm] <- count
    }
    areaData <- rbind(areaData,tempData)
  }
  
  ## Plot individual puncta data (for quality control)
  areaData$Condition = factor(areaData$Condition)
  areaData$Pre_post = factor(areaData$Pre_post)
  areaData$Roundness <- (4*pi)*(areaData$Area/areaData$Perim^2) # calculate roundness from perimeter and area
  preData = areaData[areaData$Pre_post=='PRE',]
  postData = areaData[areaData$Pre_post=='POST',]
  # Prepare plot objects
  plot1 <- list()
  plot2 <- list()
  plot3 <- list()
  # Plot area of all puncta
  p <- ggdensity(preData,
              x = "Area",
              add = "mean", rug = TRUE,
              color = "Condition", fill = "Condition",
              palette = c("#00AFBB", "#E7B800")) + 
      labs(title = 'Presynaptic puncta area', 
           x = "Area (μm^2)", y = "Density")
  plot1[[1]] <- p
  p <- ggdensity(postData,
            x = "Area",
            add = "mean", rug = TRUE,
            color = "Condition", fill = "Condition",
            palette = c("#00AFBB", "#E7B800")) + 
    labs(title = 'Postsynaptic puncta area', 
         x = "Area (μm^2)", y = "Density")
  plot1[[2]] <- p
  image=grid.arrange(grobs = plot1, ncol = 1)
  ggsave(file=sprintf("%sArea_puncta_distributions.svg",saveDirectory),
         plot=image,width=10,height=8)
  # Plot perimeter of all puncta
  p <- ggdensity(preData,
                        x = "Perim",
                        add = "mean", rug = TRUE,
                        color = "Condition", fill = "Condition",
                        palette = c("#00AFBB", "#E7B800")) + 
    labs(title = 'Presynaptic puncta perimeter', 
         x = "Perimeter (μm)", y = "Density")
  plot2[[1]] <- p
  p <- ggdensity(postData,
                        x = "Perim",
                        add = "mean", rug = TRUE,
                        color = "Condition", fill = "Condition",
                        palette = c("#00AFBB", "#E7B800")) + 
    labs(title = 'Postsynaptic puncta perimeter', 
         x = "Perimeter (μm)", y = "Density")
  plot2[[2]] <- p
  image=grid.arrange(grobs = plot2, ncol = 1)
  ggsave(file=sprintf("%sPerimeter_puncta_distributions.svg",saveDirectory),
         plot=image,width=10,height=8)
  # Plot roundness of all puncta
  p <- ggdensity(preData,
                        x = "Roundness",
                        add = "mean", rug = TRUE,
                        color = "Condition", fill = "Condition",
                        palette = c("#00AFBB", "#E7B800")) + 
    labs(title = 'Presynaptic puncta roundness', 
         x = "Roundness", y = "Density")
  plot3[[1]] <- p
  p <- ggdensity(postData,
                        x = "Roundness",
                        add = "mean", rug = TRUE,
                        color = "Condition", fill = "Condition",
                        palette = c("#00AFBB", "#E7B800")) + 
    labs(title = 'Postsynaptic puncta roundness', 
         x = "Roundness", y = "Density")
  plot3[[2]] <- p
  image = grid.arrange(grobs = plot3, ncol = 1)
  ggsave(file=sprintf("%sRoundness_puncta_distributions.svg",saveDirectory),
         plot=image,width=10,height=8)
  
  # Create data frame
  data_df <- data.frame(
    Mouse = unlist(mouse),
    Frequency = frequency,
    Condition = unlist(condition),
    Presynaptic_N = presynapticN,
    Postsynaptic_N = postsynapticN,
    Colocalised_N = colocalisedN,
    N_cells = nCells,
    Presynaptic_P = presynapticN / nCells,
    Postsynaptic_P = postsynapticN / nCells,
    Colocalised_P = colocalisedN / nCells,
    Mean_area_presynaptic = meanAreaPre,
    Mean_area_postsynaptic = meanAreaPost
  )
  
  # Convert to tibble for better printing
  data_tbl <- as_tibble(data_df)
  
  # Create plots
  conditions <- unique(data_tbl$Condition)
  mice <- unique(data_tbl$Mouse)
  frequencies <- unique(data_tbl$Frequency)
  
  # Generate random colors for each mouse
  mouseColours <- data.frame(
    Mouse = mice,
    Colour = I(replicate(length(mice), rgb(runif(1), runif(1), runif(1)))))
    
    conditionCols <- c("b", "r")
    
    # Plot 1: N puncta
    plot_list1 <- list()
    colNumbers <- c(4, 5, 6)
    
    for (count in seq_along(colNumbers)) {
      col <- colNumbers[count]
      p <- ggplot(data_tbl, aes(x = factor(Frequency), 
                                y = .data[[names(data_tbl)[col]]],
                                fill = factor(Condition))) +
        geom_boxplot() +
        scale_fill_manual(values=c("#ef8a62","#67a9cf")) +
        geom_point(position=position_dodge(width=0.75),
                   aes(group=factor(Condition))) +
        labs(title = names(data_tbl)[col], 
             x = "Frequency (kHz)", y = "Count") +
        stat_compare_means(aes(group = factor(Condition)),
                           method = 'kruskal.test',
                           label = "p.format",
                           label.y = max(data_tbl[[names(data_tbl)[col]]]+5),
                           size=3) + 
        ylim(0,max(data_tbl[[names(data_tbl)[col]]])+10) +
        theme_minimal()

      plot_list1[[count]] <- p
    }
    
    image = grid.arrange(grobs = plot_list1, ncol = 1)
    ggsave(file=sprintf("%sN_puncta_boxplots.svg",saveDirectory),
           plot=image,width=10,height=8)
    
    # Plot 2: N puncta / N cells
    plot_list2 <- list()
    colNumbers <- c(8, 9, 10)
    
    for (count in seq_along(colNumbers)) {
      col <- colNumbers[count]
      p <- ggplot(data_tbl, aes(x = factor(Frequency), 
                                y = .data[[names(data_tbl)[col]]],
                                fill = factor(Condition))) +
        geom_boxplot() +
        scale_fill_manual(values=c("#ef8a62","#67a9cf")) +
        geom_point(position=position_dodge(width=0.75),
                   aes(group=factor(Condition))) +
        labs(title = names(data_tbl)[col], 
             x = "Frequency (kHz)", y = "nPuncta / nIHCs") +
        stat_compare_means(aes(group = factor(Condition)),
                           method = 'kruskal.test',
                           label = "p.format",
                           label.y = max(data_tbl[[names(data_tbl)[col]]]+2),
                           size=3) + 
        ylim(0,max(data_tbl[[names(data_tbl)[col]]])+4) +
        theme_minimal()

      plot_list2[[count]] <- p
    }
    
    image = grid.arrange(grobs = plot_list2, ncol = 1)
    ggsave(file=sprintf("%sPuncta_per_cell_boxplots.svg",saveDirectory),
           plot=image,width=10,height=8)
    
    write.csv(data_tbl,sprintf('%sPuncta_datatable.csv',saveDirectory)) # save the datatable
