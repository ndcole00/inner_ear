preprocess_puncta_data <- function() {
  

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
  
  # Set file paths
  dataFile <- '/home/nick/Documents/Analysis/MATLAB/Puncta_analysis/Data'
  cellCountFile <- '/home/nick/Documents/Data/Blinded_data/Cell_counts.csv'
  
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
  colocalisedN <- numeric()
  mm <- 0
  
  # Process each file
  for (ii in seq_along(files)) {
    # Skip the first two files (usually "." and ".." in MATLAB)
    if (ii <= 2) next
    
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
          roisToRemove[overlap$Var1] <- TRUE
        }
      }
    }
    
    # Remove marked ROIs
    data <- data[!roisToRemove, ]
    
    # Count puncta based on file type
    if (grepl("CTBP", files[ii])) {
      presynapticN[mm] <- nrow(data)
    } else if (grepl("Homer", files[ii])) {
      postsynapticN[mm] <- nrow(data)
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
  }
  
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
    Colocalised_P = colocalisedN / nCells
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
        geom_point(position=position_dodge(width=0.75),
                   aes(group=factor(Condition))) +
        geom_dotplot(binaxis = "y", stackdir = "center", dotsize = 0.5) +
        scale_color_manual(values = c(conditionCols, mouseColours$Colour)) +
        labs(title = names(data_tbl)[col], 
             x = "Frequency (kHz)", y = "Count") +
        theme_minimal() +
        theme(axis.text.x = element_text(hjust = 1))
      
      plot_list1[[count]] <- p
    }
    
    grid.arrange(grobs = plot_list1, ncol = 1)
    
    # Plot 2: N puncta / N cells
    plot_list2 <- list()
    colNumbers <- c(8, 9, 10)
    
    for (count in seq_along(colNumbers)) {
      col <- colNumbers[count]
      p <- ggplot(data_tbl, aes(x = factor(Frequency), 
                                y = .data[[names(data_tbl)[col]]],
                                fill = factor(Condition))) +
        geom_boxplot() +
        geom_point(position=position_dodge(width=0.75),
                   aes(group=factor(Condition))) +
        scale_color_manual(values = c(conditionCols, mouseColours$Colour)) +
        labs(title = names(data_tbl)[col], 
             x = "Frequency (kHz)", y = "Count") +
        theme_minimal() +
        theme(axis.text.x = element_text(hjust = 1))
      
      plot_list2[[count]] <- p
    }
    
    grid.arrange(grobs = plot_list2, ncol = 1)
    
    return(data_tbl)
}
