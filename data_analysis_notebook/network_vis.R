library(network)
library(ggplot2)
library(GGally)
options(help_type="text")
library(sna)

library(base)
library(rstudioapi)

get_directory <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file <- "--file="
  rstudio <- "RStudio"
  
  match <- grep(rstudio, args)
  if (length(match) > 0) {
    return(dirname(rstudioapi::getSourceEditorContext()$path))
  } else {
    match <- grep(file, args)
    if (length(match) > 0) {
      return(dirname(normalizePath(sub(file, "", args[match]))))
    } else {
      return(dirname(normalizePath(sys.frames()[[1]]$ofile)))
    }
  }
}

generate_colour_list <- function(adj_mat_data) {
  species_names <- adj_mat_data[, 1]
  N_col <- "steelblue"
  AHL_col <- "darkorchid1"
  mic_col <- "tomato"
  
  colours <- c()
  i = 1
  for (species in species_names) {
    species_colour <- NULL
    
    if (grepl('N_', species)) {
      species_colour <- N_col
    }
    
    if (grepl('A_', species)) {
      species_colour <- AHL_col
    }
    
    
    if (grepl('B_', species)) {
      species_colour <- mic_col
    }
    
    colours <- c(colours, species_colour)
    i  = i + 1
  }
  
  return(colours)
}

generate_edge_colour_list <- function(directed_network, adj_mat) {
  edge_list <- as.edgelist(directed_network)
  edge_count <- network.edgecount(directed_network)
  
  negative_edge_colour <- "red"
  positive_edge_colour <- "lightgreen"
  
  edge_colours <- c()
  
  n_rows <- dim(adj_mat)[1]
  n_cols <- dim(adj_mat)[2]
  
  for (from_idx in seq(1, n_cols)) {
    for (to_idx in seq(1, n_rows)) {
      value <- adj_mat[to_idx, from_idx]
      if (value > 0) {
        edge_colours <- c(edge_colours, positive_edge_colour)
      }
      
      if (value < 0) {
        edge_colours <- c(edge_colours, negative_edge_colour)
      }
      
    }
  }

  return(edge_colours)
}

get_positive_matrix <- function(adj_mat) {
  n_rows <- dim(adj_mat)[1]
  n_cols <- dim(adj_mat)[2]
  for(idx_row in seq(from=1, to=n_rows)) {
    for(idx_col in seq(from=1, to=n_cols)) {
      val <- abs(adj_mat[idx_row, idx_col])
      adj_mat[idx_row, idx_col] <- val
    }
  }

  return(adj_mat)
}

visualise_network <- function(adj_mat_path) {
  adj_mat_data <- read.csv(adj_mat_path, header=TRUE)
  species_colours <- generate_colour_list(adj_mat_data)
  original_adj_mat <- as.matrix(adj_mat_data)
  original_adj_mat <- t(original_adj_mat)
  
  original_adj_mat <- original_adj_mat[-(1),]
  class(original_adj_mat) <- "numeric"
  
  all_positive_adj_mat <- get_positive_matrix(original_adj_mat)

  net = network(all_positive_adj_mat, directed=TRUE, edge.color="blue", labels=c(rep("z", 6)))
  edge_colours <- generate_edge_colour_list(net, original_adj_mat)
  
  species_names <- adj_mat_data[, 1]
  
  set.edge.attribute(net, "color", edge_colours)
  
  size_node <- 20
  ggnet_obj <- ggnet2(net, size=size_node, label=species_names, alpha=0.5, color=species_colours, 
                      edge.alpha=0.4, edge.color=edge_colours, edge.size = 1.5,
                      arrow.size = 10, arrow.gap = 0.060, arrow.type = "closed")

  return(ggnet_obj)
}


# adjacency_mat_dir <- "/home/behzad/Documents/barnes_lab/sympy_consortium_framework/output/two_species_no_symm/adj_matricies"
adjacency_mat_dir <- "/home/behzad/Documents/barnes_lab/sympy_consortium_framework/output/3_species_space/adj_matricies"

files <- list.files(path=adjacency_mat_dir, pattern="*.csv", full.names=TRUE, recursive=FALSE)
this_dir <- get_directory()

for (f in files) {
    system_net <- visualise_network(f)
    out_dir <-  "/networks_output/three_species/"
    out_name <- basename(f)
    print(out_name)
    out_name <- tools::file_path_sans_ext(out_name)
    # out_name <- paste(wd, out_dir, out_name, ".pdf", sep="")
    pdf(paste(this_dir, out_dir, out_name, ".pdf", sep=""))
    print(system_net)
    dev.off()
  }
